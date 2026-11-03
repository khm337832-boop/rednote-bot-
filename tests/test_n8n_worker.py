import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

# Importing bot has module-level configuration, so make the test environment
# explicit and network-free before the import.
os.environ["BOT_TOKEN"] = ""
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["WORKER_API_KEY"] = "test-worker-key"
os.environ["WHISPER_API_KEY"] = "test-whisper-key"
os.environ["PIPELINE_MODE"] = "direct"
os.environ["RECAP_FLOW"] = "manual"
os.environ["N8N_WEBHOOK_URL"] = ""
os.environ["N8N_WEBHOOK_SECRET"] = ""
os.environ["RENDER_EXTERNAL_URL"] = "https://worker.example"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx
import numpy as np

import bot


MALE_PRONOUN = "\u1000\u103b\u103d\u1014\u103a\u1010\u1031\u102c\u103a"
FEMALE_PRONOUN = "\u1000\u103b\u103d\u1014\u103a\u1019"


def first_person_text(length: int) -> str:
    return (MALE_PRONOUN + ("\u1000" * length))[:length]


class WorkerPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rednote-worker-tests-")
        self.tmp_path = Path(self.tmp.name)
        self.jobs_dir = self.tmp_path / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.patchers = []

        def start(patcher):
            self.patchers.append(patcher)
            return patcher.start()

        self.start_patch = start
        start(patch.object(bot, "WORK_ROOT", str(self.tmp_path)))
        start(patch.object(bot, "WORKER_JOBS_DIR", str(self.jobs_dir)))
        start(patch.object(bot, "BOT_TOKEN", ""))
        start(patch.object(bot, "GEMINI_API_KEY", "test-gemini-key"))
        start(patch.object(bot, "WHISPER_API_KEY", "test-whisper-key"))
        start(patch.object(bot, "WORKER_API_KEY", "test-worker-key"))
        start(patch.object(bot, "PIPELINE_MODE", "direct"))
        start(patch.object(bot, "RECAP_FLOW", "manual"))
        start(patch.object(bot, "MANUAL_PARTS_ENABLED", True))
        start(patch.object(bot, "N8N_WEBHOOK_URL", ""))
        start(patch.object(bot, "N8N_WEBHOOK_SECRET", ""))

        self.active_jobs = start(patch.object(bot, "ACTIVE_JOBS", {}))
        self.pending_links = start(patch.object(bot, "PENDING_LINKS", {}))
        self.pending_inputs = start(patch.object(bot, "PENDING_INPUTS", {}))
        self.worker_tasks = start(patch.object(bot, "WORKER_TASKS", {}))
        self.manual_sessions_by_chat = start(
            patch.object(bot, "MANUAL_SESSIONS_BY_CHAT", {})
        )
        self.manual_render_tasks = start(
            patch.object(bot, "MANUAL_RENDER_TASKS", {})
        )
        start(patch.object(bot, "MANUAL_SESSION_LOCK", asyncio.Lock()))
        self.cancelled_jobs = start(patch.object(bot, "CANCELLED_JOBS", set()))
        self.background_tasks = start(
            patch.object(bot, "BACKGROUND_TASKS", set())
        )
        self.persistent_keyboard_chats = start(
            patch.object(bot, "PERSISTENT_KEYBOARD_CHATS", set())
        )
        start(patch.object(bot, "WORKER_JOB_LOCK", asyncio.Lock()))
        start(patch.object(bot, "WORKER_STAGE_SEMAPHORE", asyncio.Semaphore(1)))
        start(patch.object(bot, "video_render_semaphore", asyncio.Semaphore(1)))

        self.mock_send_text = start(
            patch.object(bot, "send_text", new=AsyncMock(return_value=999))
        )
        self.mock_edit_text = start(
            patch.object(bot, "edit_text", new=AsyncMock(return_value=None))
        )
        self.mock_tg_api = start(
            patch.object(bot, "tg_api", new=AsyncMock(return_value={"ok": True}))
        )
        self.mock_tg_upload = start(
            patch.object(
                bot, "tg_upload_media", new=AsyncMock(return_value={"ok": True})
            )
        )
        self.mock_bump_stat = start(patch.object(bot, "bump_stat", new=Mock()))

        # The learned script-length rate and the last TTS provider are
        # process-global; reset both so tests never inherit a rate from a
        # previous test (or a real measurement) and stay deterministic.
        start(patch.object(bot, "_LAST_TTS_PROVIDER", ""))
        start(patch.object(bot, "TTS_PROVIDER_ORDER", ("edge", "gemini")))
        start(patch.dict(bot._LEARNED_CHARS_PER_SEC, {}, clear=True))
        # TTS diagnostics are process-global too; start every test clean so
        # a previous test's provider error cannot leak into /healthz asserts.
        start(patch.object(bot, "_LAST_TTS_ERROR", ""))
        start(patch.object(bot, "_LAST_TTS_SILENCED_SCENES", 0))
        start(patch.object(bot, "_LAST_TTS_RETRY_RECOVERED", 0))
        start(patch.dict(bot._LAST_TTS_ERROR_BY_MODEL, {}, clear=True))

        if hasattr(bot, "USER_SETTINGS"):
            bot.USER_SETTINGS.clear()

    async def asyncTearDown(self):
        tasks = [
            task
            for task in (
                list(bot.WORKER_TASKS.values())
                + list(bot.MANUAL_RENDER_TASKS.values())
                + list(bot.BACKGROUND_TASKS)
            )
            if not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        bot.WORKER_TASKS.clear()
        bot.MANUAL_RENDER_TASKS.clear()
        bot.MANUAL_SESSIONS_BY_CHAT.clear()
        bot.ACTIVE_JOBS.clear()
        bot.PENDING_LINKS.clear()
        bot.PENDING_INPUTS.clear()
        bot.CANCELLED_JOBS.clear()
        bot.BACKGROUND_TASKS.clear()
        bot.PERSISTENT_KEYBOARD_CHATS.clear()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.tmp.cleanup()

    @property
    def auth_headers(self):
        return {"X-Worker-Key": "test-worker-key"}

    @contextlib.asynccontextmanager
    async def api_client(self):
        transport = httpx.ASGITransport(app=bot.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client

    def create_job(
        self,
        job_id="job-test-1",
        chat_id=10001,
        url="https://videos.example/watch/1",
        settings=None,
        status_message_id=None,
    ):
        return bot._create_worker_job(
            job_id=job_id,
            chat_id=chat_id,
            url=url,
            reply_msg_id=321,
            status_message_id=status_message_id,
            settings_override=settings,
        )

    def write_job_json(self, state, relative_path, data):
        path = Path(state["job_dir"]) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def install_fake_stage(self, stage_name="download", wait=False):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def fake_stage(state, tracker=None):
            calls.append(state["job_id"])
            started.set()
            if wait:
                await release.wait()

        original = bot._worker_stage_function
        self.start_patch(
            patch.object(
                bot,
                "_worker_stage_function",
                lambda stage: fake_stage if stage == stage_name else original(stage),
            )
        )
        return started, release, calls

    async def wait_for_stage_status(self, job_id, stage, expected):
        for _ in range(200):
            state = bot._load_worker_job(job_id)
            if state["stages"][stage]["status"] == expected:
                return state
            await asyncio.sleep(0.01)
        self.fail(f"stage {stage} did not reach {expected}")

    def test_grouping_and_coverage_are_worker_owned_and_sorted(self):
        segments = [
            {"id": 2, "start": 5, "end": 12.5, "text": " second "},
            {"id": 1, "start": 0, "end": 5, "text": "first"},
            {"id": 99, "start": 2, "end": 2, "text": "invalid duration"},
            {"id": 98, "start": 3, "end": 4, "text": "   "},
            {"id": 3, "start": 15, "end": 20, "text": "third"},
            {"id": 4, "start": 21, "end": 28, "text": "fourth"},
            "not-a-dict",
        ]

        groups = bot._group_transcript_segments(segments)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["source_segment_ids"], [1, 2])
        self.assertEqual(groups[0]["start"], 0.0)
        self.assertEqual(groups[0]["end"], 12.5)
        self.assertEqual(groups[0]["speech_seconds"], 12.5)
        self.assertEqual(groups[1]["source_segment_ids"], [3, 4])
        self.assertEqual(groups[1]["speech_seconds"], 12.0)

        coverage = bot._calculate_transcript_coverage(groups, {1: "covered"})
        self.assertEqual(coverage["speech_seconds_total"], 24.5)
        self.assertEqual(coverage["speech_seconds_covered"], 12.5)
        self.assertAlmostEqual(coverage["speech_coverage_ratio"], 0.5102, places=4)
        self.assertEqual(coverage["segment_count_total"], 4)
        self.assertEqual(coverage["segment_count_covered"], 2)
        self.assertEqual(coverage["missing_group_ids"], [2])

    def test_rewrite_parser_accepts_only_allowed_nonempty_groups(self):
        scenes, protagonist, pronoun = bot._parse_rewrite_response(
            {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": [
                    {"group_id": 1, "burmese": "  first  "},
                    {"group_id": 2, "text": "second"},
                    {"group_id": 999, "burmese": "ignored"},
                    {"group_id": "bad", "burmese": "ignored"},
                    {"group_id": 1, "burmese": "   "},
                ],
            },
            [1, 2],
        )
        self.assertEqual(scenes, {1: "first", 2: "second"})
        self.assertEqual(protagonist, "Hero")
        self.assertEqual(pronoun, MALE_PRONOUN)

        scenes, _, pronoun = bot._parse_rewrite_response(
            [{"group_id": 2, "burmese": "list form"}], [2]
        )
        self.assertEqual(scenes, {2: "list form"})
        self.assertEqual(pronoun, "")

    async def test_rewrite_retries_missing_groups_and_tops_up_short_scenes(self):
        state = self.create_job(
            job_id="job-rewrite-1",
            chat_id=10002,
            settings={"coverage": 90, "rate": 10.0, "voice": "Charon", "pov": "third"},
        )
        self.assertEqual(state["settings"]["pov"], "first")
        self.write_job_json(
            state,
            "transcript.json",
            {
                "language": "en",
                "segments": [
                    {"id": 11, "start": 0, "end": 12, "text": "opening facts"},
                    {"id": 22, "start": 20, "end": 32, "text": "later facts"},
                ],
            },
        )

        calls = []

        async def fake_gemini(prompt, client=None):
            calls.append(prompt)
            if len(calls) == 1:
                return {
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                    "scenes": [{"group_id": 1, "burmese": first_person_text(40)}],
                }
            if len(calls) == 2:
                return {
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                    "scenes": [{"group_id": 2, "burmese": first_person_text(40)}],
                }
            return {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": [
                    {"group_id": 1, "burmese": first_person_text(108)},
                    {"group_id": 2, "burmese": first_person_text(108)},
                ],
            }

        self.start_patch(
            patch.object(bot, "genai", SimpleNamespace(Client=lambda api_key: object()))
        )
        self.start_patch(patch.object(bot, "_worker_gemini_json", new=fake_gemini))
        self.start_patch(
            patch.object(bot, "extract_story_summary", new=AsyncMock(return_value={}))
        )

        await bot._worker_stage_rewrite(state)

        self.assertEqual(len(calls), 3)
        script = json.loads(
            (Path(state["job_dir"]) / "script.json").read_text(encoding="utf-8")
        )
        self.assertEqual(script["coverage"]["speech_coverage_ratio"], 1.0)
        self.assertEqual(script["coverage"]["target_burmese_chars"], 216)
        self.assertEqual(script["coverage"]["actual_burmese_chars"], 216)
        self.assertEqual([s["group_id"] for s in script["scenes"]], [1, 2])
        self.assertEqual(script["scenes"][0]["source_segment_ids"], [11])
        self.assertEqual(script["scenes"][1]["source_segment_ids"], [22])
        self.assertEqual(script["scenes"][0]["start"], 0.0)
        self.assertEqual(script["scenes"][1]["end"], 32.0)
        self.assertEqual(state["stages"]["rewrite"]["status"], "done")

    async def test_rewrite_rejects_transcript_coverage_below_selected_target(self):
        state = self.create_job(
            job_id="job-rewrite-2",
            chat_id=10003,
            settings={"coverage": 90, "rate": 10.0},
        )
        self.write_job_json(
            state,
            "transcript.json",
            {
                "segments": [
                    {"id": 1, "start": 0, "end": 12, "text": "only covered"},
                    {"id": 2, "start": 20, "end": 32, "text": "never covered"},
                ]
            },
        )

        async def fake_gemini(prompt, client=None):
            return {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": [{"group_id": 1, "burmese": first_person_text(216)}],
            }

        self.start_patch(
            patch.object(bot, "genai", SimpleNamespace(Client=lambda api_key: object()))
        )
        self.start_patch(patch.object(bot, "_worker_gemini_json", new=fake_gemini))
        self.start_patch(
            patch.object(bot, "extract_story_summary", new=AsyncMock(return_value={}))
        )

        with self.assertRaises(bot.WorkerJobError):
            await bot._worker_stage_rewrite(state)
        self.assertFalse((Path(state["job_dir"]) / "script.json").exists())

    async def test_rewrite_rejects_present_but_far_too_short_scenes(self):
        state = self.create_job(
            job_id="job-rewrite-too-short",
            chat_id=10008,
            settings={"coverage": 50, "rate": 10.0},
        )
        # A stale/user-selected value below 90 is raised to the global floor.
        self.assertEqual(state["settings"]["coverage"], 90)
        self.write_job_json(
            state,
            "transcript.json",
            {
                "segments": [
                    {"id": 1, "start": 0, "end": 12, "text": "first long passage"},
                    {"id": 2, "start": 20, "end": 32, "text": "second long passage"},
                ]
            },
        )

        async def fake_gemini(prompt, client=None):
            return {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": [
                    {"group_id": 1, "burmese": first_person_text(20)},
                    {"group_id": 2, "burmese": first_person_text(20)},
                ],
            }

        self.start_patch(
            patch.object(bot, "genai", SimpleNamespace(Client=lambda api_key: object()))
        )
        self.start_patch(patch.object(bot, "_worker_gemini_json", new=fake_gemini))
        self.start_patch(
            patch.object(bot, "extract_story_summary", new=AsyncMock(return_value={}))
        )

        with self.assertRaises(bot.WorkerJobError):
            await bot._worker_stage_rewrite(state)
        self.assertFalse((Path(state["job_dir"]) / "script.json").exists())

    async def test_rewrite_targets_source_duration_not_only_active_speech(self):
        state = self.create_job(
            job_id="job-rewrite-source-duration",
            chat_id=10009,
            settings={"coverage": 90, "rate": 10.0},
        )
        self.write_job_json(
            state,
            "transcript.json",
            {
                "source_duration_seconds": 100,
                "speech_duration_seconds": 24,
                "segments": [
                    {"id": 1, "start": 0, "end": 12, "text": "opening dialogue"},
                    {"id": 2, "start": 88, "end": 100, "text": "closing dialogue"},
                ],
            },
        )

        async def fake_gemini(prompt, client=None):
            return {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": [
                    {"group_id": 1, "burmese": first_person_text(792)},
                    {"group_id": 2, "burmese": first_person_text(108)},
                ],
            }

        self.start_patch(
            patch.object(bot, "genai", SimpleNamespace(Client=lambda api_key: object()))
        )
        self.start_patch(patch.object(bot, "_worker_gemini_json", new=fake_gemini))
        self.start_patch(
            patch.object(bot, "extract_story_summary", new=AsyncMock(return_value={}))
        )

        await bot._worker_stage_rewrite(state)

        script = json.loads(
            (Path(state["job_dir"]) / "script.json").read_text(encoding="utf-8")
        )
        self.assertEqual(script["coverage"]["target_basis"], "source_duration")
        self.assertEqual(script["coverage"]["target_basis_seconds"], 100.0)
        self.assertEqual(script["coverage"]["target_narration_seconds"], 90.0)
        self.assertEqual(script["coverage"]["target_burmese_chars"], 900)
        self.assertEqual(script["coverage"]["actual_burmese_chars"], 900)
        self.assertEqual(script["target_by_group"], {"1": 792, "2": 108})
        self.assertEqual(
            [scene["timeline_seconds"] for scene in script["scenes"]],
            [88.0, 12.0],
        )

    async def test_rewrite_rejects_transcript_truncated_to_first_33_seconds(self):
        state = self.create_job(
            job_id="job-rewrite-truncated-transcript",
            chat_id=10011,
            settings={"coverage": 90, "rate": 10.0},
        )
        self.write_job_json(
            state,
            "transcript.json",
            {
                "source_duration_seconds": 540,
                "speech_duration_seconds": 33,
                "segments": [
                    {"id": 1, "start": 0, "end": 33, "text": "opening only"}
                ],
            },
        )

        with self.assertRaises(bot.WorkerJobError):
            await bot._worker_stage_rewrite(state)
        self.assertFalse((Path(state["job_dir"]) / "script.json").exists())

    def test_required_coverage_floor_and_one_liner_coverage(self):
        self.assertEqual(bot._with_required_coverage({"coverage": 50})["coverage"], 90)
        self.assertEqual(bot._with_required_coverage({"coverage": 95})["coverage"], 95)
        self.assertEqual(bot._with_required_coverage({"coverage": "bad"})["coverage"], 90)
        with patch.dict(bot._PERSISTED_USERS, {"987654321": {"coverage": 50}}):
            self.assertEqual(bot.get_user_settings(987654321)["coverage"], 90)

        groups = [
            {
                "group_id": 1,
                "source_segment_ids": [1],
                "speech_seconds": 10,
            },
            {
                "group_id": 2,
                "source_segment_ids": [2],
                "speech_seconds": 10,
            },
        ]
        coverage = bot._calculate_transcript_coverage(
            groups,
            {1: "x" * 100, 2: "short"},
            target_by_group={1: 100, 2: 100},
        )
        self.assertEqual(coverage["speech_seconds_covered"], 10.0)
        self.assertEqual(coverage["missing_group_ids"], [])
        self.assertEqual(coverage["underfilled_group_ids"], [2])

    def test_source_is_always_planned_in_two_minute_parts(self):
        three_hours = bot.build_source_segment_plan(3 * 60 * 60)
        self.assertEqual(len(three_hours), 90)
        self.assertTrue(all(item["duration"] == 120 for item in three_hours))
        self.assertEqual(three_hours[0]["start"], 0)
        self.assertEqual(three_hours[-1]["end"], 3 * 60 * 60)

        with_remainder = bot.build_source_segment_plan(301)
        self.assertEqual(
            [(x["start"], x["end"], x["duration"]) for x in with_remainder],
            [(0, 120, 120), (120, 240, 120), (240, 301, 61)],
        )
        self.assertEqual(bot.build_source_segment_plan(0), [])

        # A stale 10-minute setting from an older bot version cannot change
        # the fixed two-minute intake architecture.
        settings = bot._with_required_coverage({"chunk": 600})
        self.assertEqual(settings["chunk"], bot.SOURCE_SEGMENT_SECONDS)
        self.assertEqual(bot.DEFAULT_SETTINGS["chunk"], 120)

    def test_telegram_video_attachment_detection(self):
        telegram_video = bot._telegram_video_attachment(
            {
                "video": {
                    "file_id": "video-file",
                    "file_unique_id": "unique-1",
                    "duration": 180,
                    "file_size": 123456,
                    "mime_type": "video/mp4",
                }
            }
        )
        self.assertEqual(telegram_video["file_id"], "video-file")
        self.assertEqual(telegram_video["extension"], ".mp4")

        document_video = bot._telegram_video_attachment(
            {
                "document": {
                    "file_id": "doc-file",
                    "file_name": "../../movie.mkv",
                    "mime_type": "video/x-matroska",
                }
            }
        )
        self.assertEqual(document_video["file_name"], "movie.mkv")
        self.assertEqual(document_video["extension"], ".mkv")

        self.assertIsNone(
            bot._telegram_video_attachment(
                {
                    "document": {
                        "file_id": "pdf-file",
                        "file_name": "notes.pdf",
                        "mime_type": "application/pdf",
                    }
                }
            )
        )

    async def test_download_telegram_file_streams_to_disk(self):
        get_file = AsyncMock(
            return_value={
                "ok": True,
                "result": {"file_path": "videos/file_1.mp4"},
            }
        )
        self.start_patch(patch.object(bot, "tg_api", new=get_file))
        self.start_patch(patch.object(bot, "BOT_TOKEN", "test-token"))
        self.start_patch(
            patch.object(bot, "probe_video_info", new=Mock(return_value=(12.0, 0, 0)))
        )
        requested = {}

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def aiter_bytes(self, _size):
                yield b"0123456789" * 1100

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def stream(self, method, url):
                requested["method"] = method
                requested["url"] = url
                return FakeResponse()

        self.start_patch(patch.object(bot.httpx, "AsyncClient", FakeClient))
        out_path = self.tmp_path / "telegram-source.mp4"
        duration = await bot.download_telegram_file("file-1", str(out_path))

        self.assertEqual(duration, 12.0)
        self.assertEqual(out_path.read_bytes(), b"0123456789" * 1100)
        self.assertEqual(requested["method"], "GET")
        self.assertIn("/file/bottest-token/videos/file_1.mp4", requested["url"])

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    async def test_prepare_source_segments_creates_ordered_parts(self):
        source_path = self.tmp_path / "five-second-source.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN, "-y",
                "-f", "lavfi",
                "-i", "color=c=blue:s=64x64:r=10:d=5",
                "-g", "5",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(source_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        work_dir = self.tmp_path / "segment-job"
        work_dir.mkdir()
        segments = await bot._prepare_source_segments(
            str(source_path), str(work_dir), 5, segment_seconds=2
        )

        self.assertEqual([item["index"] for item in segments], [1, 2, 3])
        self.assertTrue(all(Path(item["path"]).exists() for item in segments))
        self.assertAlmostEqual(sum(item["duration"] for item in segments), 5, delta=0.4)
        manifest = json.loads((work_dir / "source_segments.json").read_text())
        self.assertEqual(manifest["source_segment_seconds"], 2)
        self.assertEqual(len(manifest["segments"]), 3)

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    async def test_prepare_source_segments_falls_back_when_keyframes_are_sparse(self):
        source_path = self.tmp_path / "sparse-keyframe-source.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN, "-y",
                "-f", "lavfi",
                "-i", "color=c=red:s=64x64:r=10:d=5",
                "-g", "1000",
                "-keyint_min", "1000",
                "-sc_threshold", "0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(source_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        work_dir = self.tmp_path / "sparse-segment-job"
        work_dir.mkdir()
        segments = await bot._prepare_source_segments(
            str(source_path), str(work_dir), 5, segment_seconds=2
        )

        self.assertEqual([item["index"] for item in segments], [1, 2, 3])
        self.assertTrue(all(item["duration"] <= 2.25 for item in segments))
        self.assertAlmostEqual(sum(item["duration"] for item in segments), 5, delta=0.4)

    async def test_prepare_source_segments_reports_progress_while_splitting(self):
        work_dir = self.tmp_path / "segment-progress"
        work_dir.mkdir()
        updates = []

        async def fake_update(text, force=False):
            updates.append(text)

        tracker = SimpleNamespace(update=fake_update)

        def fake_split(_path, out_dir, _segment_seconds):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            results = []
            for index in range(3):
                part = out / f"part_{index:04d}.mp4"
                part.write_bytes(b"x" * 100)
                results.append(str(part))
                time.sleep(0.05)
            return results

        def fake_probe(path):
            return (1.0 if str(path).endswith("part_0002.mp4") else 2.0, 64, 64)

        self.start_patch(patch.object(bot, "_split_video_once", new=fake_split))
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))

        segments = await bot._prepare_source_segments(
            str(self.tmp_path / "source.mp4"),
            str(work_dir),
            5,
            segment_seconds=2,
            tracker=tracker,
            progress_interval=0.01,
        )

        self.assertEqual([item["index"] for item in segments], [1, 2, 3])
        self.assertTrue(any("ခွဲနေဆဲ" in text and "file" in text for text in updates))
        self.assertTrue(any("duration စစ်နေ" in text for text in updates))
        manifest = json.loads((work_dir / "source_segments.json").read_text())
        self.assertEqual(manifest["split_mode"], "stream_copy")

    async def test_prepare_source_segments_reports_fallback_progress(self):
        work_dir = self.tmp_path / "segment-fallback-progress"
        work_dir.mkdir()
        updates = []

        async def fake_update(text, force=False):
            updates.append(text)

        tracker = SimpleNamespace(update=fake_update)

        def fake_extract(_source, output_path, _start, _duration, **_kwargs):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 50)
            time.sleep(0.03)
            with path.open("ab") as fh:
                fh.write(b"x" * 50)
            return True

        def fake_probe(path):
            return (1.0 if str(path).endswith("0003.mp4") else 2.0, 64, 64)

        self.start_patch(patch.object(bot, "_split_video_once", return_value=[]))
        self.start_patch(
            patch.object(bot, "_extract_source_segment_file", new=fake_extract)
        )
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))

        segments = await bot._prepare_source_segments(
            str(self.tmp_path / "source.mp4"),
            str(work_dir),
            5,
            segment_seconds=2,
            tracker=tracker,
            progress_interval=0.01,
        )

        self.assertEqual([item["index"] for item in segments], [1, 2, 3])
        self.assertTrue(any("တိကျစွာ ခွဲနေသည်" in text for text in updates))
        self.assertTrue(any("ခွဲပြီး အပိုင်း 3/3" in text for text in updates))
        manifest = json.loads((work_dir / "source_segments.json").read_text())
        self.assertEqual(manifest["split_mode"], "accurate_reencode")

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    def test_concat_video_parts_joins_recap_files_in_order(self):
        parts = []
        for index, color in enumerate(("blue", "green"), start=1):
            part = self.tmp_path / f"recap-{index}.mp4"
            result = subprocess.run(
                [
                    bot.FFMPEG_BIN, "-y",
                    "-f", "lavfi",
                    "-i", f"color=c={color}:s=64x64:r=10:d=1",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(part),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                self.skipTest("could not generate a local MP4 fixture")
            parts.append(str(part))

        output = self.tmp_path / "joined.mp4"
        concat_list = self.tmp_path / "joined.txt"
        self.assertTrue(
            bot._concat_video_parts(parts, str(output), str(concat_list))
        )
        duration, _, _ = bot.probe_video_info(str(output))
        self.assertAlmostEqual(duration, 2.0, delta=0.25)
        self.assertEqual(
            concat_list.read_text(encoding="utf-8").count("file '"), 2
        )

    def test_final_duration_window_is_literal_source_percentage(self):
        settings = {"coverage": 90, "rate": 10.0, "speed": 2.0}
        target, low, high = bot._final_duration_window(600, settings)

        # A 10-minute source must produce a 9-minute final.  Voice speed
        # changes how much text is needed, never the requested final time.
        # v5.18: the window is ±3% of the source (text-driven trimming can
        # only move in whole words, ~1-3s, so ±2% was unreachable and made
        # calibration oscillate long <-> short).
        self.assertEqual((target, low, high), (540.0, 522.0, 558.0))
        self.assertEqual(bot.script_target_chars(100, settings), 1800)
        self.assertTrue(bot._duration_is_on_target(540, 600, settings))
        self.assertFalse(bot._duration_is_on_target(33, 600, settings))
        self.assertFalse(bot._duration_is_on_target(599, 600, settings))
        self.assertEqual(bot._duration_status(33, 600, settings), "short")
        self.assertEqual(bot._duration_status(599, 600, settings), "long")

        # v5.15: trimming is sentence-granular (see the dedicated
        # sentence-boundary tests), so it lands close to the requested
        # budget instead of cutting an artificial unpunctuated run to the
        # exact character -- and it never leaves a half sentence behind.
        scenes = [
            {
                "group_id": 1,
                "start": 0,
                "burmese": (
                    MALE_PRONOUN
                    + " ဇာတ်လမ်းကို အစကနေ ပြန်ပြောပြမယ်။ "
                    "မနက်စောစော အိမ်ကနေ ထွက်လာခဲ့တယ်။"
                ),
            },
            {
                "group_id": 2,
                "start": 10,
                "burmese": (
                    MALE_PRONOUN
                    + " ရုံးမှာ အလုပ်တွေ အများကြီး ရှိခဲ့တယ်။ "
                    "ညနေမှာ အိမ်ပြန်ရောက်ခဲ့တယ်။"
                ),
            },
        ]
        original_total = bot.script_total_chars(scenes)
        trimmed = bot.trim_scenes_proportionally(scenes, int(original_total * 0.6))
        self.assertTrue(trimmed)
        self.assertLess(bot.script_total_chars(trimmed), original_total)
        for scene in trimmed:
            self.assertTrue(
                bot._scene_is_first_person(scene["burmese"], MALE_PRONOUN)
            )
            self.assertIn(
                scene["burmese"].strip()[-1], bot.SCRIPT_SENTENCE_END_CHARS
            )

    # --- script length is planned from the voice's measured rate ---
    def test_script_length_rate_is_learned_not_hardcoded_6_5(self):
        # A fresh process on the default edge-first order plans with the
        # cold-start seed for the Edge voice, then replaces it with the
        # first real measurement.  Production measured my-MM-ThihaNeural
        # at ~14-17 chars/s (a 313-char scene took 18.65s); the earlier
        # 2.3 seed -- inferred from an oversized script's total, not
        # from the voice -- mis-sized every cold-start script ~6x short
        # and sent calibration into expand/trim oscillation.
        bot.reset_learned_chars_per_second()
        self.assertEqual(bot.TTS_PROVIDER_ORDER[0], "edge")
        self.assertAlmostEqual(bot.learned_chars_per_second(), 12.0, places=6)
        self.assertAlmostEqual(bot._settings_script_rate({}), 12.0, places=6)

        # The shipped 6.5 default means "learn it for me"; any other value
        # is a deliberate user override and is kept.
        self.assertAlmostEqual(
            bot._settings_script_rate({"rate": 6.5}), 12.0, places=6
        )
        self.assertAlmostEqual(
            bot._settings_script_rate({"rate": 7.5}), 7.5, places=6
        )

        # A 120s window at 90% is 1296 chars at the 12 chars/s seed --
        # within one calibration round of the voice's real speed, so the
        # first script no longer needs heavy expansion *or* trimming.
        self.assertEqual(bot.script_target_chars(120, {}), 1296)
        self.assertEqual(bot._rewrite_target_chars_for_duration(120, {}), 1296)
        # Gemini seeds at its own measured ballpark.
        self.assertAlmostEqual(
            bot.learned_chars_per_second("gemini"), 10.0,
            places=6,
        )

    def test_narration_measurements_teach_the_learned_rate(self):
        scenes = [
            {"start": i * 12, "burmese": first_person_text(30)} for i in range(5)
        ]
        clips = [
            np.full(int(12 * bot.SAMPLE_RATE), 0.5, dtype=np.float32)
            for _ in range(5)
        ]
        # 150 script chars over 60s of real audio = 2.5 chars/second
        measurements = bot._narration_measurements(scenes, clips, {"pause": 0})
        self.assertAlmostEqual(
            measurements["effective_chars_per_second"], 2.5, places=3
        )
        learned_edge = bot.learned_chars_per_second("edge")
        # The first real measurement replaces the cold-start seed
        # outright (no EMA from a guessed seed).
        self.assertAlmostEqual(learned_edge, 2.5, places=3)
        self.assertAlmostEqual(
            bot.learned_chars_per_second_by_provider()["edge"],
            learned_edge,
            places=3,
        )
        # The next window/job is sized with what this voice actually did.
        self.assertEqual(
            bot.script_target_chars(120, {}), int(120 * 0.9 * learned_edge)
        )

        # Rates are learned per provider: Gemini's own measurements never
        # overwrite the Edge estimate.
        bot._LAST_TTS_PROVIDER = "gemini"
        gemini_measurements = bot._narration_measurements(
            scenes, clips[:3] + clips[:2], {"pause": 0}
        )
        self.assertAlmostEqual(
            bot.learned_chars_per_second("gemini"),
            gemini_measurements["effective_chars_per_second"],
            places=3,
        )
        self.assertAlmostEqual(
            bot.learned_chars_per_second("edge"), learned_edge, places=3
        )

    def test_learned_rate_is_clamped_and_ignores_tiny_audio(self):
        self.assertEqual(bot.SCRIPT_CHARS_PER_SEC_MIN, 1.2)
        # The ceiling must clear the real Edge voice (~14-17 chars/s
        # measured in production): the old 9.0 cap pinned the learned
        # rate well under the voice's actual speed.
        self.assertEqual(bot.SCRIPT_CHARS_PER_SEC_MAX, 24.0)
        for _ in range(8):
            bot.record_narration_rate({
                "speech_seconds": 10.0,
                "spoken_chars_measured": 500,
                "effective_chars_per_second": 50.0,
            })
        self.assertAlmostEqual(
            bot.learned_chars_per_second("edge"),
            bot.SCRIPT_CHARS_PER_SEC_MAX,
            places=6,
        )

        bot.reset_learned_chars_per_second()
        for _ in range(8):
            bot.record_narration_rate({
                "speech_seconds": 10.0,
                "spoken_chars_measured": 500,
                "effective_chars_per_second": 0.2,
            })
        self.assertAlmostEqual(
            bot.learned_chars_per_second("edge"),
            bot.SCRIPT_CHARS_PER_SEC_MIN,
            places=6,
        )

        # A partial/short clip is not a trustworthy measurement.
        bot.reset_learned_chars_per_second()
        bot.record_narration_rate({
            "speech_seconds": 0.5,
            "spoken_chars_measured": 900,
            "effective_chars_per_second": 8.0,
        })
        self.assertAlmostEqual(
            bot.learned_chars_per_second("edge"), 12.0, places=6
        )

    def test_trim_cuts_only_at_sentence_boundaries(self):
        sentences = [
            MALE_PRONOUN + " မနက်စောစော အိမ်ကနေ ထွက်လာခဲ့တယ်။",
            "လမ်းမှာ မိုးက သည်းသည်းမည်းမည်း ရွာနေတယ်။",
            "ကျွန်တော် ထီးမပါလို့ တစ်ကိုယ်လုံး စိုးရွှဲသွားတယ်။",
            "ဒါပေမယ့် ကျွန်တော် ရုံးကို အချိန်မီ ရောက်ခဲ့တယ်။",
            "ညနေမှာ ကျွန်တော် သူငယ်ချင်းနဲ့ ပြန်ဆုံဖြစ်တယ်။",
            "အဲဒီနောက် ကျွန်တော်တို့ အတူ ထမင်းစားဖြစ်တယ်။",
        ]
        # Each scene opens with a first-person sentence (the prompt rule),
        # so trimming trailing sentences can never lose the pronoun.
        scenes = [
            {"start": 0, "burmese": "".join(sentences[0:3])},
            {"start": 30, "burmese": "".join(sentences[2:4])},
            {"start": 60, "burmese": "".join(sentences[4:6])},
        ]
        source_texts = [scene["burmese"] for scene in scenes]
        original_total = bot.script_total_chars(scenes)

        trimmed = bot.trim_scenes_proportionally(
            scenes, int(original_total * 0.6)
        )

        self.assertTrue(trimmed)
        trimmed_total = bot.script_total_chars(trimmed)
        self.assertLess(trimmed_total, original_total)
        for scene in trimmed:
            text = scene["burmese"].strip()
            # The result always ends on a terminator so the TTS does not
            # read a flat, unfinished fragment.
            self.assertIn(text[-1], bot.SCRIPT_SENTENCE_END_CHARS)
            # Only a leading part of the source scene survives: whole
            # sentences first, and (v5.18) the last few characters may be
            # cut at a word boundary -- never in the middle of a word and
            # never leaving a dangling combining mark behind.
            core = text
            if not any(source.startswith(core) for source in source_texts):
                core = core[:-1].rstrip()
                self.assertTrue(
                    any(source.startswith(core) for source in source_texts),
                    f"trimmed text is not a prefix of its source: {text!r}",
                )
            self.assertNotIn(core[-1], bot._BURMESE_TRAILING_JUNK)
            self.assertTrue(bot._scene_is_first_person(text, MALE_PRONOUN))
            for sentence in bot._split_sentences(core):
                self.assertTrue(
                    any(sentence in source for source in source_texts)
                )
        # Every surviving sentence that is not the trimmed tail is complete.
        for source in source_texts:
            for sentence in bot._split_sentences(source):
                if sentence in "".join(scene["burmese"] for scene in trimmed):
                    self.assertIn(sentence[-1], bot.SCRIPT_SENTENCE_END_CHARS)

    def test_trim_drops_a_whole_scene_instead_of_chopping_it(self):
        single = MALE_PRONOUN + " ဒီအကြောင်းကို အစကနေ အဆုံးထိ ပြန်ပြောပြမယ်။"
        # A budget below the only sentence removes the text (the scene is
        # then dropped by trim_scenes_proportionally) instead of shipping
        # half a clause to the TTS.
        self.assertEqual(bot._trim_text_to_char_budget(single, 10), "")
        self.assertEqual(
            bot._trim_text_to_char_budget(single, len(single)), single
        )

        scenes = [
            {"start": 0, "burmese": single},
            {
                "start": 10,
                "burmese": (
                    MALE_PRONOUN + " ရုံးမှာ အစည်းအေး တစ်ခု လုပ်ဖြစ်တယ်။ "
                    "အဲဒီမှာ ကျွန်တော် သူနဲ့ ပြန်ဆုံဖြစ်တယ်။"
                ),
            },
        ]
        trimmed = bot.trim_scenes_proportionally(
            scenes, int(bot.script_total_chars(scenes) * 0.45)
        )
        # Nothing in the result is a chopped fragment.
        for scene in trimmed:
            text = scene["burmese"].strip()
            self.assertIn(text[-1], bot.SCRIPT_SENTENCE_END_CHARS)
        self.assertGreaterEqual(bot.script_total_chars(trimmed), 1)

    def test_unpunctuated_text_falls_back_to_a_word_boundary(self):
        unstructured = MALE_PRONOUN + " ဒီအကြောင်းကို အစကနေ အဆုံးထိ"
        cut = bot._trim_text_to_char_budget(unstructured, 20)
        self.assertLessEqual(len(cut), 20)
        self.assertTrue(cut)
        self.assertTrue(bot._scene_is_first_person(cut, MALE_PRONOUN))

    def test_first_person_validation_uses_actual_scene_text(self):
        possessive = "ကျွန်တော့်ရဲ့ အတွေ့အကြုံကို အစကနေပြန်ပြောပါမယ်။"
        opposite = FEMALE_PRONOUN + " အစကနေပြန်ပြောပါမယ်။"
        mixed = first_person_text(30) + opposite

        self.assertTrue(bot._scene_is_first_person(possessive, MALE_PRONOUN))
        self.assertFalse(bot._scene_is_first_person(opposite, MALE_PRONOUN))
        self.assertFalse(bot._scene_is_first_person(mixed, MALE_PRONOUN))
        self.assertFalse(
            bot._scene_is_first_person("သူ အိမ်ပြန်သွားတယ်။", MALE_PRONOUN)
        )

        metrics = bot._first_person_metrics(
            [possessive, "သူ အိမ်ပြန်သွားတယ်။"], MALE_PRONOUN
        )
        self.assertEqual(metrics["bad_scene_indexes"], [1])
        self.assertFalse(metrics["all_scenes_first_person"])

    def test_direct_gemini_json_parser_recovers_wrapped_and_truncated_scenes(self):
        fenced = "```json\n" + json.dumps(
            {
                "scenes": [
                    {
                        "start": 0,
                        "end": 120,
                        "burmese": first_person_text(702),
                    }
                ]
            },
            ensure_ascii=False,
        ) + "\n```"
        parsed_fenced = bot._parse_gemini_json_text(fenced)
        self.assertEqual(len(parsed_fenced["scenes"]), 1)

        truncated = (
            '{"scenes": ['
            + json.dumps(
                {"start": 0, "end": 60, "burmese": first_person_text(351)},
                ensure_ascii=False,
            )
            + ", "
            + json.dumps(
                {"start": 60, "end": 120, "burmese": first_person_text(351)},
                ensure_ascii=False,
            )
            + ', {"start": 120, "burmese": "unfinished'
        )
        parsed_truncated = bot._parse_gemini_json_text(truncated)
        self.assertEqual(len(parsed_truncated["scenes"]), 2)

        normalized = bot._normalize_direct_script_payload(
            {
                "result": {
                    "main_character": {"name": "Hero"},
                    "first_person_pronoun": MALE_PRONOUN,
                    "segments": [
                        {
                            "start_time": "0",
                            "end_time": "120",
                            "text": first_person_text(702),
                        }
                    ],
                }
            },
            chunk_duration=120,
        )
        self.assertEqual(normalized["protagonist"], "Hero")
        self.assertEqual(normalized["pronoun"], MALE_PRONOUN)
        self.assertEqual(normalized["scenes"][0]["start"], 0.0)
        self.assertEqual(normalized["scenes"][0]["end"], 120.0)
        self.assertEqual(len(normalized["scenes"][0]["burmese"]), 702)

    async def test_worker_first_person_repair_rewrites_third_person_scene(self):
        groups = [
            {
                "group_id": 1,
                "source_segment_ids": [1],
                "start": 0,
                "end": 10,
                "speech_seconds": 10,
                "timeline_seconds": 10,
                "source_text": "The hero walks home and opens the door.",
            }
        ]
        scenes_by_group = {1: "သူ အိမ်ပြန်သွားပြီး တံခါးဖွင့်လိုက်တယ်။" * 4}
        story_context = {"pronoun": MALE_PRONOUN, "protagonist": "Hero"}

        async def fake_gemini(prompt, client):
            self.assertIn("FIRST-PERSON REPAIR", prompt)
            return {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": [
                    {"group_id": 1, "burmese": first_person_text(200)}
                ],
            }

        self.start_patch(
            patch.object(bot, "_worker_gemini_json", new=fake_gemini)
        )
        selected, bad = await bot._repair_worker_first_person_scenes(
            groups,
            scenes_by_group,
            {1: 100},
            story_context,
            {"coverage": 90, "voice": "Charon"},
            object(),
            rounds=1,
        )

        self.assertEqual(selected, MALE_PRONOUN)
        self.assertEqual(bad, [])
        self.assertTrue(
            bot._scene_is_first_person(scenes_by_group[1], MALE_PRONOUN)
        )

    async def test_worker_duration_adjustment_accepts_first_person_expansion(self):
        script = {
            "story_context": {"pronoun": MALE_PRONOUN, "protagonist": "Hero"},
            "groups": [
                {
                    "group_id": 1,
                    "source_segment_ids": [1],
                    "start": 0,
                    "end": 10,
                    "speech_seconds": 10,
                    "timeline_seconds": 10,
                    "source_text": "The hero walks home and opens the door.",
                }
            ],
        }
        scenes = [
            {
                "group_id": 1,
                "start": 0,
                "end": 10,
                "burmese": first_person_text(100),
            }
        ]

        async def fake_gemini(prompt, client):
            self.assertIn('"desired_burmese_chars": 200', prompt)
            return {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": [
                    {"group_id": 1, "burmese": first_person_text(200)}
                ],
            }

        self.start_patch(
            patch.object(bot.genai, "Client", lambda api_key: object())
        )
        self.start_patch(
            patch.object(bot, "_worker_gemini_json", new=fake_gemini)
        )

        updated, changed, selected = await bot._worker_adjust_scenes_for_duration(
            {"job_id": "adjust-test", "chat_id": 1},
            script,
            scenes,
            {1: 200},
            "expand",
            {"coverage": 90, "voice": "Charon"},
        )

        self.assertEqual(selected, MALE_PRONOUN)
        self.assertEqual(changed, [1])
        self.assertEqual(len(updated[0]["burmese"]), 200)

    async def test_direct_recap_rejects_script_far_below_global_90_target(self):
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/test-video",
        )
        response_payload = {
            "protagonist": "Hero",
            "pronoun": MALE_PRONOUN,
            "scenes": [
                {
                    "start": 0,
                    "end": 100,
                    "burmese": first_person_text(20),
                }
            ],
            "cuts": [],
        }
        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(
                generate_content=lambda model, contents, config: SimpleNamespace(
                    text=json.dumps(response_payload, ensure_ascii=False)
                )
            ),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        with self.assertRaises(RuntimeError) as raised:
            await bot.build_chunk_recap(
                str(self.tmp_path / "chunk.mp4"),
                100,
                0,
                100,
                {},
                1,
                tracker,
                settings={"coverage": 50, "rate": 10.0},
            )
        self.assertIn("/900 chars", str(raised.exception))

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    async def test_direct_recap_repairs_first_person_and_renders_90_percent(self):
        # This test counts generate calls to verify the recap logic;
        # pin racing off (covered separately) for a deterministic 1:1
        # call mapping.
        self.start_patch(
            patch.object(bot, "GEMINI_HEDGE_INITIAL_PARALLEL", 1)
        )
        chunk_path = self.tmp_path / "direct_chunk.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64:r=25:d=3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(chunk_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/test-video",
        )
        third_person = "သူ " + ("က" * 24)
        repaired = first_person_text(27)
        generate_calls = []

        def fake_generate(model, contents, config):
            # Prompt is the last string part (repair calls are text-only
            # now, so it may be the only part).
            prompt = next(
                (str(c) for c in reversed(contents) if isinstance(c, str)),
                "",
            )
            generate_calls.append(prompt)
            if "FIRST-PERSON REPAIR ONLY" in prompt:
                payload = {
                    "pronoun": MALE_PRONOUN,
                    "scenes": [
                        {
                            "scene_index": 0,
                            "start": 0,
                            "end": 3,
                            "burmese": repaired,
                        }
                    ],
                }
            elif "ယခင် scenes များကို ပြန်မပေးပါနှင့်" in prompt:
                payload = {
                    "pronoun": MALE_PRONOUN,
                    "scenes": [
                        {
                            "start": 1.5,
                            "end": 3,
                            "burmese": first_person_text(28),
                        }
                    ],
                    "cuts": [],
                }
            else:
                payload = {
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                    "scenes": [
                        {"start": 0, "end": 3, "burmese": third_person}
                    ],
                    "cuts": [],
                }
            return SimpleNamespace(
                text=json.dumps(payload, ensure_ascii=False)
            )

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=fake_generate),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        async def fake_direct_tts(tts_scenes, work_dir, settings):
            # Simulate a voice that actually speaks at 20 chars/sec while
            # the initial estimate was 10: the first 27-char script is only
            # 1.35s, so build_chunk_recap must expand and synthesize again.
            return [
                np.full(
                    max(1, int(bot.SAMPLE_RATE * len(scene["burmese"]) / 20.0)),
                    0.5,
                    dtype=np.float32,
                )
                for scene in tts_scenes
            ]

        self.start_patch(
            patch.object(bot, "synthesize_tts", new=fake_direct_tts)
        )
        self.start_patch(
            patch.object(
                bot,
                "extract_story_summary",
                new=AsyncMock(
                    return_value={
                        "summary": "done",
                        "characters": [],
                        "last_scene": "end",
                        "protagonist": "Hero",
                        "pronoun": MALE_PRONOUN,
                    }
                ),
            )
        )
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        out_chunk, summary = await bot.build_chunk_recap(
            str(chunk_path),
            3,
            0,
            3,
            {},
            1,
            tracker,
            settings={
                "coverage": 90,
                "rate": 10.0,
                "speed": 1.0,
                "pause": 0,
                "voice": "Charon",
                "resolution": 64,
                "shield": False,
                "color": False,
                "quality": 35,
                "fast_mode": True,
            },
        )

        self.assertEqual(summary["pronoun"], MALE_PRONOUN)
        self.assertEqual(len(generate_calls), 3)
        # Narration is spoken: every script prompt must demand the
        # spoken register (user: written-style script sounds like
        # reading a book aloud).
        self.assertTrue(
            all("စကားပြောဟန်" in p for p in generate_calls),
            generate_calls,
        )
        self.assertIn("FIRST-PERSON REPAIR ONLY", generate_calls[1])
        self.assertIn("ယခင် scenes များကို ပြန်မပေးပါနှင့်", generate_calls[2])
        final_duration, _, _ = bot.probe_video_info(out_chunk)
        self.assertAlmostEqual(final_duration, 2.75, delta=0.2)

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    # --- v5.18: measured-duration calibration (production 124.2s/113s) ---
    def test_duration_trim_lands_inside_the_window_from_measured_rate(self):
        """Regression: a 125.5s window measuring 124.2s against a 113s target
        used to fail with "refusing to render a video with the wrong
        duration" after three full attempts.

        One whole Burmese sentence is ~10-15s of speech, so a
        sentence-only trim jumped from "long" to "short".  Trimming to the
        character budget implied by the *measured* rate (with the tail cut
        at a word boundary) has to land inside the window.
        """
        settings = dict(bot.DEFAULT_SETTINGS)
        source_seconds = 125.5
        target, low, high = bot._final_duration_window(source_seconds, settings)
        pause = bot._settings_pause_seconds(settings)
        sentence = "ကျွန်တော် ဒီနေ့ မနက်စာ စားပြီး အလုပ်သွားတယ်။"
        scenes = [
            {"start": index * 12.5, "end": (index + 1) * 12.5, "burmese": sentence}
            for index in range(10)
        ]
        measured_seconds = 124.2
        total_chars = bot.script_total_chars(scenes)
        rate = total_chars / measured_seconds
        measurements = {
            "speech_seconds": measured_seconds,
            "pause_seconds": pause,
            "pause_total_seconds": pause * (len(scenes) - 1),
            "narration_seconds": measured_seconds + pause * (len(scenes) - 1),
            "measured_chars_per_second": rate,
            "effective_chars_per_second": rate,
        }

        trimmed = bot._trim_scenes_for_duration(scenes, measurements, target)

        self.assertLess(
            bot.script_total_chars(trimmed), total_chars, "nothing was trimmed"
        )
        predicted = (
            bot.script_total_chars(trimmed) / rate
            + pause * (len(trimmed) - 1)
        )
        self.assertTrue(
            low <= predicted <= high,
            f"trimmed narration predicts {predicted:.1f}s, "
            f"window is {low:.1f}-{high:.1f}s",
        )
        # Only the model's own words survive, in order.
        for scene in trimmed:
            text = scene["burmese"].strip()
            self.assertTrue(sentence.startswith(text.rstrip("။")))
            self.assertIn(text[-1], bot.SCRIPT_SENTENCE_END_CHARS)
            self.assertNotIn(
                text[-2] if len(text) > 1 else " ", bot._BURMESE_TRAILING_JUNK
            )

    def test_duration_trim_is_a_noop_without_a_measured_rate(self):
        scenes = [
            {"start": 0, "burmese": "ကျွန်တော် ပထမ scene ပါ။"},
            {"start": 10, "burmese": "ကျွန်တော် ဒုတိယ scene ပါ။"},
        ]
        self.assertEqual(
            bot._trim_scenes_for_duration(scenes, {}, 113.0), scenes
        )
        self.assertEqual(bot._trim_scenes_for_duration([], {}, 113.0), [])

    def test_word_tail_cut_keeps_words_whole_and_endings_readable(self):
        text = "ကျွန်တော် မနက်စာ စားပြီး အလုပ်သွားတယ်။ ညနေမှာ အိမ်ပြန်ခဲ့တယ်။"
        source = text

        trimmed = bot.trim_scenes_proportionally(
            [{"start": 0, "burmese": text}, {"start": 20, "burmese": text}],
            int(bot.script_total_chars([{"burmese": text}]) * 1.2),
        )

        self.assertTrue(trimmed)
        for scene in trimmed:
            value = scene["burmese"].strip()
            self.assertTrue(value)
            self.assertIn(value[-1], bot.SCRIPT_SENTENCE_END_CHARS)
            core = value[:-1] if value[-1] in bot.SCRIPT_SENTENCE_END_CHARS else value
            self.assertTrue(source.startswith(core), core)
            self.assertNotIn(core[-1], bot._BURMESE_TRAILING_JUNK)

    # --- v5.19: a job must never die silently (restart / stall) ---
    def _job_state_file(self):
        path = str(self.tmp_path / "active_jobs.json")
        self.start_patch(patch.object(bot, "JOB_STATE_FILE", path))
        self.addCleanup(bot.ACTIVE_JOB_STATE.clear)
        self.addCleanup(bot.LAST_JOB_RESULTS.clear)
        self.addCleanup(bot.ACTIVE_JOBS.clear)
        self.addCleanup(setattr, bot, "_LAST_INTERRUPTED_JOBS", [])
        return path

    async def test_restart_recovery_tells_the_user_their_job_died(self):
        """Production: /status said "nothing is running" while the user's
        recap was mid-render, because a deploy killed the process.  The
        next process must say so instead of silently forgetting it."""
        state_file = self._job_state_file()
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "active": {
                        "777": {
                            "job_id": "jdead01",
                            "stage": "video ဖန်တီးနေသည်",
                            "started_at": time.time() - 600,
                            "updated_at": time.time() - 540,
                        }
                    },
                    "results": {"777": {"status": "done", "at": time.time() - 9000}},
                },
                f,
                ensure_ascii=False,
            )
        sent = []

        async def fake_send_text(chat_id, text, reply_to=None, reply_markup=None):
            sent.append((chat_id, text))
            return 1

        self.start_patch(patch.object(bot, "send_text", new=fake_send_text))

        recovered = await bot._recover_interrupted_jobs()

        self.assertEqual(recovered, 1)
        self.assertEqual(len(sent), 1)
        chat_id, text = sent[0]
        self.assertEqual(chat_id, 777)
        self.assertIn("restart", text)
        self.assertIn("video ဖန်တီးနေသည်", text)
        self.assertIn("ပြန်ပို့", text)
        self.assertEqual(
            bot.LAST_JOB_RESULTS["777"]["status"], "interrupted"
        )
        self.assertEqual(bot.ACTIVE_JOB_STATE, {})
        self.assertEqual(bot._LAST_INTERRUPTED_JOBS[0]["chat_id"], "777")
        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved.get("active"), {})
        self.assertIn("777", saved.get("results") or {})

        # The status screen now explains it instead of a bare "idle".
        status_text = bot.job_status_text(777)
        self.assertIn("လုပ်နေတာ မရှိပါ", status_text)
        self.assertIn("server restart", status_text)

    async def test_watchdog_releases_a_stalled_job_and_keeps_live_ones(self):
        self._job_state_file()
        sent = []

        async def fake_send_text(chat_id, text, reply_to=None, reply_markup=None):
            sent.append((chat_id, text))
            return 1

        self.start_patch(patch.object(bot, "send_text", new=fake_send_text))
        now = time.time()
        bot.ACTIVE_JOBS[1] = "jstalled"
        bot.register_active_job(1, "jstalled", stage="Gemini ဆီ ဗီဒီယိုတင်ပြီး")
        bot.ACTIVE_JOB_STATE["1"]["updated_at"] = now - (bot.JOB_STALL_SECONDS + 30)
        bot.ACTIVE_JOBS[2] = "jlive"
        bot.register_active_job(2, "jlive", stage="အသံသွင်းနေသည်")

        released = await bot._job_watchdog_tick()

        self.assertEqual(released, 1)
        self.assertNotIn(1, bot.ACTIVE_JOBS)
        self.assertIn(2, bot.ACTIVE_JOBS)
        self.assertEqual(bot.LAST_JOB_RESULTS["1"]["status"], "failed")
        self.assertIn("stalled", bot.LAST_JOB_RESULTS["1"]["error"])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], 1)
        self.assertIn("ရပ်လိုက်ပါတယ်", sent[0][1])

    async def test_active_job_state_is_mirrored_to_disk(self):
        state_file = self._job_state_file()

        bot.register_active_job(5, "jdisk01", stage="script ရေးနေသည်")
        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["active"]["5"]["job_id"], "jdisk01")
        self.assertEqual(saved["active"]["5"]["stage"], "script ရေးနေသည်")

        bot.touch_active_job(5, stage="အသံသွင်းနေသည်")
        self.assertEqual(bot.ACTIVE_JOB_STATE["5"]["stage"], "အသံသွင်းနေသည်")

        bot.finish_active_job(5, "failed", "Measured TTS duration is outside")
        self.assertEqual(bot.ACTIVE_JOB_STATE, {})
        self.assertEqual(bot.LAST_JOB_RESULTS["5"]["status"], "failed")
        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["active"], {})
        self.assertIn("5", saved["results"])

        # A dead registry entry must never survive as a phantom job that
        # blocks the chat forever.
        self.assertNotIn(5, bot.ACTIVE_JOBS)

    def test_status_text_shows_the_live_stage(self):
        self._job_state_file()
        bot.ACTIVE_JOBS[9] = "jlive9"
        bot.register_active_job(9, "jlive9", stage="အသံသွင်းနေသည်")
        text = bot.job_status_text(9)
        self.assertIn("jlive9", text)
        self.assertIn("အသံသွင်းနေသည်", text)

    async def test_direct_recap_sized_by_learned_rate_is_not_shredded(self):
        """v5.15 acceptance: a 2-minute window must reach ~108s because the
        script was *sized* right, not because calibration deleted two
        thirds of it (the reported 341s -> 112s job)."""
        self.start_patch(
            patch.object(bot, "GEMINI_HEDGE_INITIAL_PARALLEL", 1)
        )
        chunk_path = self.tmp_path / "learned_rate_chunk.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64:r=25:d=120",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(chunk_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        sentences = [
            MALE_PRONOUN + " မနက်စောစော အိမ်ကနေ ထွက်လာခဲ့တယ်။",
            "လမ်းမှာ မိုးက သည်းသည်းမည်းမည်း ရွာနေတယ်။",
            MALE_PRONOUN + " ထီးမပါလို့ စိုးရွှဲသွားတယ်။",
            "ရုံးကို အချိန်မီ ရောက်ခဲ့တယ်။",
            MALE_PRONOUN + " ရုံးမှာ အစည်းအေး တစ်ခု ရှိတယ်။",
            "အဲဒီမှာ သူငယ်ချင်းနဲ့ ပြန်ဆုံဖြစ်တယ်။",
            MALE_PRONOUN + " သူနဲ့ စကားအများကြီး ပြောဖြစ်တယ်။",
            "ညနေမှာ အိမ်ပြန်ရောက်ခဲ့တယ်။",
            MALE_PRONOUN + " ဒီနေ့က စိတ်ဝင်စားစရာ ကောင်းတဲ့ နေ့တစ်နေ့ ဖြစ်တယ်။",
            "အိပ်ရာဝင်တော့ ပင်ပန်းပေမယ့် ပျော်တယ်။",
        ]
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/test-video",
        )
        prompts = []

        def fake_generate(model, contents, config):
            prompt = next(
                (str(c) for c in reversed(contents) if isinstance(c, str)), ""
            )
            prompts.append(prompt)
            # A model that returns roughly the requested character count
            # in whole sentences (rotating triples from the bank so no
            # two scenes share the exact same text).
            scenes = []
            total_chars = 0
            index = 0
            while total_chars < 1350:
                text = " ".join([
                    sentences[index % len(sentences)],
                    sentences[(index + 3) % len(sentences)],
                    sentences[(index + 7) % len(sentences)],
                ])
                scenes.append({
                    "start": index * 10.0,
                    "end": (index + 1) * 10.0,
                    "burmese": text,
                })
                total_chars += len(text)
                index += 1
            payload = {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": scenes,
                "cuts": [],
            }
            return SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=fake_generate),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        tts_calls = []

        async def fake_edge_like_tts(tts_scenes, work_dir, settings):
            # Simulated at the planning seed's speed (production
            # measures the real Thiha voice at ~14-17 chars/second).
            tts_calls.append([scene["burmese"] for scene in tts_scenes])
            return [
                np.full(
                    max(1, int(bot.SAMPLE_RATE * len(scene["burmese"]) / 12.0)),
                    0.5,
                    dtype=np.float32,
                )
                for scene in tts_scenes
            ]

        self.start_patch(
            patch.object(bot, "synthesize_tts", new=fake_edge_like_tts)
        )
        self.start_patch(
            patch.object(
                bot,
                "extract_story_summary",
                new=AsyncMock(
                    return_value={
                        "summary": "done",
                        "characters": [],
                        "last_scene": "end",
                        "protagonist": "Hero",
                        "pronoun": MALE_PRONOUN,
                    }
                ),
            )
        )
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))
        settings = {
            "coverage": 90,
            "speed": 1.0,
            "pause": 0,
            "voice": "Charon",
            "resolution": 64,
            "shield": False,
            "color": False,
            "quality": 35,
            "fast_mode": True,
        }
        # The shipped 6.5 "rate" default must mean auto (learned), not 6.5.
        self.assertEqual(bot.script_target_chars(120, settings), 1296)
        self.assertEqual(bot.script_target_chars(120, {**settings, "rate": 6.5}), 1296)

        out_chunk, summary = await bot.build_chunk_recap(
            str(chunk_path), 120, 0, 120, {}, 1, tracker, settings=settings,
        )

        self.assertEqual(summary["pronoun"], MALE_PRONOUN)
        # The first script handed to TTS was already the right size for
        # the voice's measured speed, so no expand/shave round was
        # needed (long-but-inside-the-source narration is kept whole;
        # only an over-source excess is ever shaved).
        first_chars = sum(len(text) for text in tts_calls[0])
        self.assertGreaterEqual(first_chars, 1200)
        self.assertLessEqual(first_chars, 1500)
        self.assertLessEqual(len(tts_calls), 2)
        # Nothing was chopped mid-sentence on the way.
        for call in tts_calls:
            for text in call:
                self.assertIn(text.strip()[-1], bot.SCRIPT_SENTENCE_END_CHARS)
        final_duration, _, _ = bot.probe_video_info(out_chunk)
        self.assertGreaterEqual(final_duration, 120 * 0.9 * 0.98 - 0.5)
        # Between the 90% target and the source length the speech is
        # kept whole; a right-sized script lands close to target and
        # in any case never past the source itself.
        self.assertLessEqual(final_duration, 120 * 0.9 * 1.15 + 2.0)

    async def test_direct_recap_shaves_only_excess_over_source(self):
        """v5.22: a voice slower than planned must not make the recap
        longer than the source (the reported final > original job).
        Only the excess past the window is shaved -- whole sentences,
        then re-voiced -- never a trim back to the 90% target."""
        bot.reset_learned_chars_per_second()
        self.start_patch(
            patch.object(bot, "GEMINI_HEDGE_INITIAL_PARALLEL", 1)
        )
        chunk_path = self.tmp_path / "overshoot_chunk.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64:r=25:d=120",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(chunk_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        sentences = [
            MALE_PRONOUN + " မနက်စောစော အိမ်ကနေ ထွက်လာခဲ့တယ်။",
            "လမ်းမှာ မိုးက သည်းသည်းမည်းမည်း ရွာနေတယ်။",
            MALE_PRONOUN + " ထီးမပါလို့ စိုးရွှဲသွားတယ်။",
            "ရုံးကို အချိန်မီ ရောက်ခဲ့တယ်။",
            MALE_PRONOUN + " ရုံးမှာ အစည်းအေး တစ်ခု ရှိတယ်။",
            "အဲဒီမှာ သူငယ်ချင်းနဲ့ ပြန်ဆုံဖြစ်တယ်။",
            MALE_PRONOUN + " သူနဲ့ စကားအများကြီး ပြောဖြစ်တယ်။",
            "ညနေမှာ အိမ်ပြန်ရောက်ခဲ့တယ်။",
            MALE_PRONOUN + " ဒီနေ့က စိတ်ဝင်စားစရာ ကောင်းတဲ့ နေ့တစ်နေ့ ဖြစ်တယ်။",
            "အိပ်ရာဝင်တော့ ပင်ပန်းပေမယ့် ပျော်တယ်။",
        ]
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/test-video",
        )

        def fake_generate(model, contents, config):
            scenes = []
            total_chars = 0
            index = 0
            while total_chars < 1350:
                text = " ".join([
                    sentences[index % len(sentences)],
                    sentences[(index + 3) % len(sentences)],
                    sentences[(index + 7) % len(sentences)],
                ])
                scenes.append({
                    "start": index * 10.0,
                    "end": (index + 1) * 10.0,
                    "burmese": text,
                })
                total_chars += len(text)
                index += 1
            payload = {
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "scenes": scenes,
                "cuts": [],
            }
            return SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=fake_generate),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        tts_calls = []

        async def fake_slow_tts(tts_scenes, work_dir, settings):
            # Slower than the 12 chars/s planning seed: the first
            # measured narration lands well past the 120s window.
            tts_calls.append([scene["burmese"] for scene in tts_scenes])
            return [
                np.full(
                    max(1, int(bot.SAMPLE_RATE * len(scene["burmese"]) / 9.0)),
                    0.5,
                    dtype=np.float32,
                )
                for scene in tts_scenes
            ]

        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_slow_tts))
        self.start_patch(
            patch.object(
                bot,
                "extract_story_summary",
                new=AsyncMock(
                    return_value={
                        "summary": "done",
                        "characters": [],
                        "last_scene": "end",
                        "protagonist": "Hero",
                        "pronoun": MALE_PRONOUN,
                    }
                ),
            )
        )
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))
        settings = {
            "coverage": 90,
            "speed": 1.0,
            "pause": 0,
            "voice": "Charon",
            "resolution": 64,
            "shield": False,
            "color": False,
            "quality": 35,
            "fast_mode": True,
        }

        out_chunk, summary = await bot.build_chunk_recap(
            str(chunk_path), 120, 0, 120, {}, 1, tracker, settings=settings,
        )

        self.assertEqual(summary["pronoun"], MALE_PRONOUN)
        # The over-source measurement triggered a shave + re-voice.
        self.assertGreaterEqual(len(tts_calls), 2)
        first_chars = sum(len(text) for text in tts_calls[0])
        last_chars = sum(len(text) for text in tts_calls[-1])
        self.assertLess(last_chars, first_chars)
        # ...but only the excess: the kept script is still ~90% sized,
        # not shredded down toward the window floor.
        self.assertGreaterEqual(last_chars, 900)
        for call in tts_calls:
            for text in call:
                self.assertIn(text.strip()[-1], bot.SCRIPT_SENTENCE_END_CHARS)
        final_duration, _, _ = bot.probe_video_info(out_chunk)
        # The hard user rule: never longer than the source video.
        self.assertLessEqual(final_duration, 120 * 1.01)
        self.assertGreaterEqual(final_duration, 120 * 0.9 * 0.98 - 0.5)

    def test_duration_exceeds_source_and_within_limits(self):
        self.assertFalse(bot._duration_exceeds_source(100.0, 100.0))
        self.assertFalse(bot._duration_exceeds_source(101.0, 100.0))
        self.assertTrue(bot._duration_exceeds_source(103.0, 100.0))
        self.assertFalse(bot._duration_exceeds_source(500.0, 0.0))
        self.assertTrue(bot._duration_within_limits(95.0, 100.0, {}))
        self.assertTrue(bot._duration_within_limits(100.0, 100.0, {}))
        self.assertFalse(bot._duration_within_limits(104.0, 100.0, {}))
        self.assertFalse(bot._duration_within_limits(50.0, 100.0, {}))

    def test_polish_splits_runaway_sentence_at_clause_commas(self):
        clause = MALE_PRONOUN + " လမ်းလျှောက်သွားတယ်"
        runaway = "၊ ".join([clause] * 12) + "။"
        self.assertGreater(len(runaway), bot._SPEECH_MAX_SENTENCE_CHARS)
        polished = bot._polish_text_for_speech(runaway)
        sentences = bot._split_sentences(polished)
        self.assertGreaterEqual(len(sentences), 2)
        for sentence in sentences:
            self.assertLessEqual(
                len(sentence.strip()), bot._SPEECH_MAX_SENTENCE_CHARS + 40
            )
            self.assertIn(sentence.strip()[-1], bot.SCRIPT_SENTENCE_END_CHARS)
        # Words are preserved -- only punctuation/spacing may change.
        def _strip(value):
            return "".join(
                ch for ch in value if not ch.isspace() and ch not in "၊။"
            )
        self.assertEqual(_strip(polished), _strip(runaway))
        short = MALE_PRONOUN + " အိမ်ပြန်တယ်။"
        self.assertEqual(bot._polish_text_for_speech(short), short)

    def test_cap_scenes_to_source_trims_predicted_overshoot(self):
        bot.reset_learned_chars_per_second()
        sentence = MALE_PRONOUN + " အလုပ်ကို စောစောသွားတယ်။"
        scenes = [
            {
                "start": i * 40.0,
                "end": (i + 1) * 40.0,
                "burmese": " ".join([sentence] * 40),
            }
            for i in range(3)
        ]
        total_before = bot.script_total_chars(scenes)
        capped = bot._cap_scenes_to_source(scenes, 120.0, {})
        total_after = bot.script_total_chars(capped)
        self.assertLess(total_after, total_before)
        # Cap ~= 95% of 120s at the 12 chars/s cold-start seed.
        self.assertLessEqual(total_after, 1500)
        # The validated minimum is a hard floor for the cap.
        floored = bot._cap_scenes_to_source(
            scenes, 120.0, {}, minimum_chars=2500
        )
        self.assertGreaterEqual(bot.script_total_chars(floored), 2000)
        # Already-small scripts pass through untouched.
        small = [{"start": 0.0, "end": 10.0, "burmese": sentence}]
        self.assertEqual(bot._cap_scenes_to_source(small, 120.0, {}), small)

    def test_shave_scenes_to_seconds_only_shaves_excess(self):
        sentence = MALE_PRONOUN + " အလုပ်ကို စောစောသွားတယ်။"
        scenes = [
            {
                "start": i * 40.0,
                "end": (i + 1) * 40.0,
                "burmese": " ".join([sentence] * 20),
            }
            for i in range(3)
        ]
        before = bot.script_total_chars(scenes)
        measurements = {"effective_chars_per_second": 10.0, "pause_seconds": 0.0}
        shaved, changed = bot._shave_scenes_to_seconds(
            scenes, measurements, before / 10.0 / 2
        )
        self.assertTrue(changed)
        after = bot.script_total_chars(shaved)
        self.assertLess(after, before)
        self.assertGreaterEqual(after, int(before * 0.35))
        self.assertLessEqual(after, int(before * 0.65))
        # A ceiling the script already fits is a no-op.
        same, changed_again = bot._shave_scenes_to_seconds(
            shaved, measurements, 100000.0
        )
        self.assertFalse(changed_again)
        self.assertEqual(bot.script_total_chars(same), after)

    def test_dedupe_scripts_removes_cross_part_repeats(self):
        repeated = "မိုးရွာနေတယ်။"
        first = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "burmese": f"{MALE_PRONOUN} ထွက်လာတယ်။ {repeated}",
                }
            ],
            "metrics": {"target_chars": 10},
        }
        second = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "burmese": f"{repeated} {MALE_PRONOUN} ဆက်သွားတယ်။",
                }
            ],
            "metrics": {"target_chars": 10},
        }
        # Removing the repeat here would breach the part's validated
        # minimum, so it must be kept (repetition beats a summary).
        third = {
            "scenes": [
                {"start": 0.0, "end": 10.0, "burmese": repeated}
            ],
            "metrics": {"target_chars": 100000},
        }
        stats = bot._dedupe_scripts_in_order([first, second, third])
        self.assertEqual(stats["removed_sentences"], 1)
        self.assertIn(repeated, first["scenes"][0]["burmese"])
        self.assertNotIn(repeated, second["scenes"][0]["burmese"])
        self.assertIn(repeated, third["scenes"][0]["burmese"])

    async def test_update_notice_sent_once_per_version(self):
        fake_send = AsyncMock(return_value=777)
        self.start_patch(patch.object(bot, "send_text", new=fake_send))
        self.start_patch(patch.object(bot, "_save_persisted", new=lambda: None))
        users = {}
        self.start_patch(patch.object(bot, "_PERSISTED_USERS", users))
        await bot._maybe_send_update_notice(919191)
        await bot._maybe_send_update_notice(919191)
        self.assertEqual(fake_send.await_count, 1)
        self.assertEqual(users["919191"]["_seen_version"], bot.BUILD_VERSION)
        self.assertIn(bot.BUILD_VERSION, fake_send.await_args.args[1])
        # A failed send is not marked seen (retried on next contact).
        fake_send.reset_mock()
        fake_send.return_value = None
        users.clear()
        await bot._maybe_send_update_notice(919191)
        self.assertNotIn("_seen_version", users.get("919191", {}))

    def test_alignment_borrows_only_to_next_scene_and_freezes_last(self):
        scenes = [
            {"group_id": 1, "source_segment_ids": [1], "start": 0, "end": 10},
            {"group_id": 2, "source_segment_ids": [2], "start": 20, "end": 28},
            {"group_id": 3, "source_segment_ids": [3], "start": 40, "end": 45},
        ]
        plan = bot._build_alignment_plan(scenes, [2, 15, 12], 50, pause_seconds=0.2)

        self.assertEqual(plan[0]["input_seconds"], 2.2)
        self.assertEqual(plan[0]["borrowed_seconds"], 0.0)
        self.assertEqual(plan[0]["freeze_seconds"], 0.0)
        self.assertEqual(plan[0]["output_end"], 2.2)

        self.assertEqual(plan[1]["source_limit"], 40.0)
        self.assertEqual(plan[1]["input_seconds"], 15.2)
        self.assertEqual(plan[1]["borrowed_seconds"], 7.2)
        self.assertEqual(plan[1]["freeze_seconds"], 0.0)

        self.assertEqual(plan[2]["input_seconds"], 10.0)
        self.assertEqual(plan[2]["borrowed_seconds"], 5.0)
        self.assertEqual(plan[2]["freeze_seconds"], 2.0)
        self.assertEqual(plan[2]["desired_output_seconds"], 12.0)

        invalid_pause_plan = bot._build_alignment_plan(
            scenes, [1, 1, 1], 50, pause_seconds="invalid"
        )
        self.assertEqual(invalid_pause_plan[0]["desired_output_seconds"], 1.0)

        with self.assertRaises(bot.WorkerJobError):
            bot._build_alignment_plan(scenes, [1], 50)
        with self.assertRaises(bot.WorkerJobError):
            bot._build_alignment_plan(scenes, [1, 1, 1], 0)

    async def test_tts_writes_sorted_pcm_wav_and_audio_metadata(self):
        state = self.create_job(
            job_id="job-tts-1",
            chat_id=10004,
            settings={"pause": 0.1, "voice": "Charon"},
        )
        self.write_job_json(
            state,
            "script.json",
            {
                "scenes": [
                    {
                        "group_id": 2,
                        "start": 10,
                        "end": 15,
                        "burmese": first_person_text(20),
                    },
                    {
                        "group_id": 1,
                        "start": 0,
                        "end": 5,
                        "burmese": first_person_text(20),
                    },
                ]
            },
        )
        fake_tts = AsyncMock(
            return_value=[
                np.full(bot.SAMPLE_RATE, 0.5, dtype=np.float32),
                np.full(bot.SAMPLE_RATE // 2, -0.25, dtype=np.float32),
            ]
        )
        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_tts))

        await bot._worker_stage_tts(state)

        audio = json.loads(
            (Path(state["job_dir"]) / "audio.json").read_text(encoding="utf-8")
        )
        self.assertEqual([c["group_id"] for c in audio["clips"]], [1, 2])
        self.assertEqual(audio["sample_rate"], bot.SAMPLE_RATE)
        self.assertEqual(audio["total_speech_seconds"], 1.5)
        self.assertEqual(audio["total_narration_seconds"], 1.6)
        self.assertTrue((Path(state["job_dir"]) / "audio_clips/scene_0001.pcm").exists())
        with wave.open(str(Path(state["job_dir"]) / "narration.wav"), "rb") as wav:
            self.assertEqual(wav.getframerate(), bot.SAMPLE_RATE)
            self.assertEqual(wav.getnframes(), 38400)
        self.assertEqual(state["stages"]["tts"]["status"], "done")

    async def test_tts_rejects_narration_far_below_source_coverage_target(self):
        state = self.create_job(
            job_id="job-tts-too-short",
            chat_id=10010,
            settings={"coverage": 50, "pause": 0, "voice": "Charon"},
        )
        self.assertEqual(state["settings"]["coverage"], 90)
        state.setdefault("metrics", {})["source"] = {"duration_seconds": 100}
        self.write_job_json(
            state,
            "script.json",
            {
                "coverage": {
                    "source_duration_seconds": 100,
                    "target_narration_seconds": 90,
                },
                "scenes": [
                    {
                        "group_id": 1,
                        "start": 0,
                        "end": 100,
                        "burmese": first_person_text(900),
                    }
                ],
            },
        )
        fake_tts = AsyncMock(
            return_value=[np.full(bot.SAMPLE_RATE, 0.5, dtype=np.float32)]
        )
        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_tts))

        async def no_expansion(state_arg, script, scenes, targets, mode, settings, tracker=None):
            return scenes, [], MALE_PRONOUN

        self.start_patch(
            patch.object(bot, "_worker_adjust_scenes_for_duration", new=no_expansion)
        )

        with self.assertRaises(bot.WorkerJobError) as raised:
            await bot._worker_stage_tts(state)
        self.assertIn("stayed below", str(raised.exception))
        self.assertNotEqual(state["stages"]["tts"]["status"], "done")
        self.assertFalse((Path(state["job_dir"]) / "audio.json").exists())

    async def test_tts_calibrates_measured_45_seconds_to_90(self):
        state = self.create_job(
            job_id="job-tts-calibration",
            chat_id=10011,
            settings={
                "coverage": 90,
                "pause": 0,
                "rate": 10.0,
                "speed": 1.0,
                "voice": "Charon",
            },
        )
        state.setdefault("metrics", {})["source"] = {"duration_seconds": 100}
        self.write_job_json(
            state,
            "script.json",
            {
                "story_context": {
                    "pronoun": MALE_PRONOUN,
                    "protagonist": "Hero",
                },
                "coverage": {
                    "source_duration_seconds": 100,
                    "target_narration_seconds": 90,
                },
                "groups": [
                    {
                        "group_id": 1,
                        "source_segment_ids": [1],
                        "start": 0,
                        "end": 100,
                        "speech_seconds": 100,
                        "timeline_seconds": 100,
                        "source_text": "A long chronological source scene.",
                    }
                ],
                "scenes": [
                    {
                        "group_id": 1,
                        "source_segment_ids": [1],
                        "start": 0,
                        "end": 100,
                        "burmese": first_person_text(900),
                    }
                ],
            },
        )

        tts_calls = []

        async def fake_tts(scenes, work_dir, settings):
            # This job's real voice speaks at 20 chars/sec, not the initial
            # 10 chars/sec estimate: 900 chars therefore measure only 45s.
            clips = []
            for scene in scenes:
                tts_calls.append(len(scene["burmese"]))
                seconds = len(scene["burmese"]) / 20.0
                clips.append(
                    np.full(
                        max(1, int(bot.SAMPLE_RATE * seconds)),
                        0.5,
                        dtype=np.float32,
                    )
                )
            return clips

        async def fake_expand(state_arg, script, scenes, targets, mode, settings, tracker=None):
            self.assertEqual(mode, "expand")
            updated = []
            changed = []
            for scene in scenes:
                gid = int(scene["group_id"])
                new_scene = dict(scene)
                new_scene["burmese"] = first_person_text(int(targets[gid]))
                updated.append(new_scene)
                changed.append(gid)
            return updated, changed, MALE_PRONOUN

        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_tts))
        self.start_patch(
            patch.object(bot, "_worker_adjust_scenes_for_duration", new=fake_expand)
        )

        await bot._worker_stage_tts(state)

        audio = json.loads(
            (Path(state["job_dir"]) / "audio.json").read_text(encoding="utf-8")
        )
        script = json.loads(
            (Path(state["job_dir"]) / "script.json").read_text(encoding="utf-8")
        )
        # Calibration's contract is landing inside the final-duration
        # window (90s target for the 100s source), not the exact
        # center: trims intentionally aim ~1% inside the band.
        self.assertTrue(
            bot._duration_is_on_target(audio["total_narration_seconds"], 100, state["settings"]),
            audio["total_narration_seconds"],
        )
        self.assertAlmostEqual(audio["total_narration_seconds"], 90.0, delta=3.0)
        # ±3% of the 1800-character target: that is the whole band the
        # calibration is allowed to land in (v5.18 widened it from ±2%).
        self.assertAlmostEqual(
            script["coverage"]["actual_burmese_chars"], 1800, delta=54
        )
        # Two rounds are enough now: v5.18 trims by *predicted duration*
        # from the measured rate instead of by raw characters, so it hits
        # the band on the first correction instead of overshooting.
        self.assertGreaterEqual(len(audio["duration_calibration_history"]), 2)
        self.assertEqual(audio["duration_calibration_history"][0]["narration_seconds"], 45.0)
        # One TTS pass per calibration round (+1 for the expand), so two
        # rounds is exactly the fast, converging path.
        self.assertGreaterEqual(len(tts_calls), 2)
        self.assertTrue(script["coverage"]["first_person"]["all_scenes_first_person"])
        self.assertEqual(state["stages"]["tts"]["status"], "done")

    async def test_split_for_telegram_recursively_resplits_oversized_parts(self):
        source = self.tmp_path / "large.mp4"
        source.write_bytes(b"x" * 10000)
        small = self.tmp_path / "small.mp4"
        small.write_bytes(b"x" * 10)
        calls = []

        def fake_split(part, out_dir, segment_seconds):
            calls.append((str(part), out_dir, segment_seconds))
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            sizes = [2500, 2500] if len(calls) == 1 else [625, 625, 625]
            results = []
            for index, size in enumerate(sizes, start=1):
                child = out_path / f"part_{index:03d}.mp4"
                child.write_bytes(b"y" * size)
                results.append(str(child))
            return results

        self.start_patch(patch.object(bot, "MAX_TG_BYTES", 1000))
        self.start_patch(
            patch.object(bot, "probe_video_info", lambda part: (100.0, 16, 16))
        )
        self.start_patch(patch.object(bot, "_split_video_once", new=fake_split))

        self.assertEqual(
            await bot.split_for_telegram(str(small), str(self.tmp_path)), [str(small)]
        )
        parts = await bot.split_for_telegram(str(source), str(self.tmp_path))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(parts), 6)
        self.assertTrue(all(Path(part).stat().st_size <= 1000 for part in parts))

    async def test_download_stage_reuses_valid_existing_source(self):
        state = self.create_job(job_id="job-download-1", chat_id=10005)
        source = Path(state["job_dir"]) / "source.mp4"
        source.write_bytes(b"x" * 20000)
        self.start_patch(
            patch.object(bot, "probe_video_info", lambda path: (42.0, 640, 360))
        )
        self.start_patch(
            patch.object(bot, "get_source_duration", new=AsyncMock(return_value=42.0))
        )
        fake_download = AsyncMock(return_value=42.0)
        self.start_patch(patch.object(bot, "download_full", new=fake_download))

        await bot._worker_stage_download(state)

        fake_download.assert_not_awaited()
        self.assertEqual(state["stages"]["download"]["status"], "done")
        self.assertEqual(state["metrics"]["source"]["duration_seconds"], 42.0)

    async def test_deliver_stage_uploads_and_notifies(self):
        state = self.create_job(
            job_id="job-deliver-1",
            chat_id=10006,
            settings={"caption": True, "notify": True, "split_tg": True},
            status_message_id=77,
        )
        final = Path(state["job_dir"]) / "final.mp4"
        final.write_bytes(b"x" * 100)
        self.start_patch(
            patch.object(bot, "probe_video_info", lambda path: (1.25, 64, 64))
        )

        await bot._worker_stage_deliver(state)

        self.mock_tg_upload.assert_awaited_once()
        self.assertEqual(self.mock_tg_upload.await_args.args[0], "sendVideo")
        self.assertEqual(self.mock_tg_upload.await_args.args[4], str(final))
        self.assertEqual(state["stages"]["deliver"]["status"], "done")
        self.assertEqual(state["metrics"]["delivery"]["part_count"], 1)
        self.mock_send_text.assert_awaited()

    async def test_deliver_stage_refuses_33_second_final_from_100_second_source(self):
        state = self.create_job(
            job_id="job-deliver-too-short",
            chat_id=10013,
            settings={"coverage": 90, "split_tg": True},
        )
        state.setdefault("metrics", {})["source"] = {"duration_seconds": 100}
        final = Path(state["job_dir"]) / "final.mp4"
        final.write_bytes(b"x" * 100)
        self.start_patch(
            patch.object(bot, "probe_video_info", lambda path: (33.0, 64, 64))
        )

        with self.assertRaises(bot.WorkerJobError) as raised:
            await bot._worker_stage_deliver(state)

        self.assertIn("before delivery", str(raised.exception))
        self.mock_tg_upload.assert_not_awaited()
        self.assertNotEqual(state["stages"]["deliver"]["status"], "done")

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    async def test_render_stage_cuts_video_to_narration_duration(self):
        state = self.create_job(
            job_id="job-render-1",
            chat_id=10007,
            settings={
                "resolution": 64,
                "quality": 35,
                "shield": False,
                "color": False,
                "pause": 0,
                "fast_mode": True,
            },
        )
        job_dir = Path(state["job_dir"])
        source = job_dir / "source.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=25:d=3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        narration = job_dir / "narration.wav"
        samples = np.full(int(bot.SAMPLE_RATE * 2.7), 1000, dtype="<i2")
        with wave.open(str(narration), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(bot.SAMPLE_RATE)
            wav.writeframes(samples.tobytes())

        self.write_job_json(
            state,
            "script.json",
            {
                "scenes": [
                    {
                        "group_id": 1,
                        "source_segment_ids": [1],
                        "start": 0,
                        "end": 3,
                        "burmese": first_person_text(20),
                    }
                ]
            },
        )
        self.write_job_json(
            state,
            "audio.json",
            {
                "pause_seconds": 0,
                "narration_file": "narration.wav",
                "clips": [
                    {
                        "scene_index": 1,
                        "group_id": 1,
                        "duration_seconds": 2.7,
                    }
                ],
            },
        )

        await bot._worker_stage_render(state)

        final = job_dir / "final.mp4"
        self.assertTrue(final.exists() and final.stat().st_size > 0)
        final_duration, _, _ = bot.probe_video_info(str(final))
        self.assertAlmostEqual(final_duration, 2.7, delta=0.15)
        plan = json.loads((job_dir / "render_plan.json").read_text(encoding="utf-8"))["items"]
        self.assertEqual(plan[0]["desired_output_seconds"], 2.7)
        self.assertEqual(plan[0]["input_seconds"], 2.7)
        self.assertEqual(state["metrics"]["render"]["audio_alignment_filters_used"], [])

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    async def test_render_stage_rejects_final_video_far_below_90_target(self):
        state = self.create_job(
            job_id="job-render-too-short",
            chat_id=10012,
            settings={
                "resolution": 64,
                "quality": 35,
                "shield": False,
                "color": False,
                "pause": 0,
                "fast_mode": True,
            },
        )
        job_dir = Path(state["job_dir"])
        source = job_dir / "source.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:r=25:d=3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        narration = job_dir / "narration.wav"
        samples = np.full(int(bot.SAMPLE_RATE * 0.4), 1000, dtype="<i2")
        with wave.open(str(narration), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(bot.SAMPLE_RATE)
            wav.writeframes(samples.tobytes())

        self.write_job_json(
            state,
            "script.json",
            {
                "scenes": [
                    {
                        "group_id": 1,
                        "source_segment_ids": [1],
                        "start": 0,
                        "end": 3,
                        "burmese": first_person_text(20),
                    }
                ]
            },
        )
        self.write_job_json(
            state,
            "audio.json",
            {
                "pause_seconds": 0,
                "narration_file": "narration.wav",
                "clips": [
                    {
                        "scene_index": 1,
                        "group_id": 1,
                        "duration_seconds": 0.4,
                    }
                ],
            },
        )

        with self.assertRaises(bot.WorkerJobError):
            await bot._worker_stage_render(state)
        self.assertNotEqual(state["stages"]["render"]["status"], "done")

    def test_telegram_button_keyboards_are_structured_correctly(self):
        persistent = bot.persistent_menu_keyboard()
        labels = [
            button["text"]
            for row in persistent["keyboard"]
            for button in row
        ]
        self.assertTrue(persistent["is_persistent"])
        self.assertTrue(persistent["resize_keyboard"])
        self.assertEqual(
            labels,
            [
                bot.PERSISTENT_BUTTON_RECAP,
                bot.PERSISTENT_BUTTON_SETTINGS,
                bot.PERSISTENT_BUTTON_STATUS,
                bot.PERSISTENT_BUTTON_CANCEL,
                bot.PERSISTENT_BUTTON_SUMMARY,
                bot.PERSISTENT_BUTTON_HELP,
            ],
        )

        callbacks = [
            button["callback_data"]
            for row in bot.job_progress_keyboard()["inline_keyboard"]
            for button in row
        ]
        self.assertEqual(
            callbacks,
            ["menu:cancel", "menu:status", "menu:settings", "menu:main"],
        )

    async def test_start_and_persistent_buttons_route_to_real_actions(self):
        chat_id = int(bot.ADMIN_ID)

        def message_update(text):
            return {
                "message": {
                    "message_id": 10,
                    "chat": {"id": chat_id},
                    "from": {"id": chat_id},
                    "text": text,
                }
            }

        self.mock_send_text.reset_mock()
        await bot.process_admin_update(message_update("/start"))
        self.assertEqual(self.mock_send_text.await_count, 2)
        first_markup = self.mock_send_text.await_args_list[0].kwargs["reply_markup"]
        second_markup = self.mock_send_text.await_args_list[1].kwargs["reply_markup"]
        self.assertTrue(first_markup["is_persistent"])
        self.assertIn("inline_keyboard", second_markup)
        start_text = self.mock_send_text.await_args_list[1].args[1]
        for expected in (
            "Telegram ထဲ တိုက်ရိုက်ပို့တဲ့",
            "RedNote link",
            "Telegram public post link",
            "group/channel profile link",
            "၂ မိနစ်စီ",
            "၃ နာရီ = ၉၀ ပိုင်း",
            "အပိုင်းတစ်ခုချင်း",
            "အပိုင်းအားလုံးကို ကြိုမခွဲ၊ Script ကိုလည်း ကြိုမရေးပါ",
            "ပထမအပိုင်းတစ်ခုကိုပဲ ခွဲထုတ်ပြီး",
            "နောက် ၂ မိနစ်အပိုင်းတစ်ခုကိုပဲ ခွဲထုတ်ပို့",
            "🎬 Recap လုပ်မယ်",
            "Auto မ Run ဘူး",
            "20MB အောက်",
            "ဇာတ်ကောင်နာမည်တွေ",
            "ကျွန်တော်/ကျွန်မ",
            "အားလုံးကို အစဉ်အတိုင်း ပြန်ဆက်",
            "လက်ရှိ Feature ၂၀",
            "မူရင်းပိုင်းချိန်",
            "2 မိနစ်",
        ):
            self.assertIn(expected, start_text)
        self.assertLessEqual(len(start_text), 4096)

        self.mock_send_text.reset_mock()
        await bot.process_admin_update(
            message_update(bot.PERSISTENT_BUTTON_SETTINGS)
        )
        self.mock_send_text.assert_awaited_once()
        self.assertEqual(
            self.mock_send_text.await_args.kwargs["reply_markup"],
            bot.settings_keyboard(chat_id),
        )

        self.mock_send_text.reset_mock()
        await bot.process_admin_update(message_update(bot.PERSISTENT_BUTTON_RECAP))
        recap_text = self.mock_send_text.await_args.args[1]
        self.assertIn("Telegram video file", recap_text)
        self.assertIn("၂ မိနစ်စီ", recap_text)
        self.assertEqual(
            self.mock_send_text.await_args.kwargs["reply_markup"],
            bot.main_menu_keyboard(chat_id),
        )

        # A link sent to a newly deployed process before /start must also
        # install the bottom buttons once, before the job starts.
        bot.PERSISTENT_KEYBOARD_CHATS.clear()
        self.mock_send_text.reset_mock()
        fake_link_job = AsyncMock(return_value=None)
        self.start_patch(patch.object(bot, "handle_link_job", new=fake_link_job))
        await bot.process_admin_update(
            message_update("https://videos.example/watch/99")
        )
        self.mock_send_text.assert_awaited_once()
        self.assertTrue(
            self.mock_send_text.await_args.kwargs["reply_markup"]["is_persistent"]
        )
        fake_link_job.assert_awaited_once()

    async def test_uploaded_video_and_caption_link_use_the_recap_inputs(self):
        chat_id = int(bot.ADMIN_ID)
        fake_video_job = AsyncMock(return_value=None)
        self.start_patch(
            patch.object(bot, "handle_telegram_video_job", new=fake_video_job)
        )

        video_update = {
            "message": {
                "message_id": 21,
                "chat": {"id": chat_id},
                "from": {"id": chat_id},
                "video": {
                    "file_id": "uploaded-video",
                    "file_unique_id": "unique-video",
                    "duration": 755,
                    "file_size": 12_345_678,
                    "mime_type": "video/mp4",
                },
            }
        }
        bot.PERSISTENT_KEYBOARD_CHATS.clear()
        self.mock_send_text.reset_mock()
        await bot.process_admin_update(video_update)

        fake_video_job.assert_awaited_once()
        self.assertEqual(fake_video_job.await_args.args[2]["file_id"], "uploaded-video")
        self.mock_send_text.assert_awaited_once()
        self.assertTrue(
            self.mock_send_text.await_args.kwargs["reply_markup"]["is_persistent"]
        )

        # A link supplied as a caption on a photo/media message should use
        # the same link path instead of being ignored because text is empty.
        fake_link_job = AsyncMock(return_value=None)
        self.start_patch(patch.object(bot, "handle_link_job", new=fake_link_job))
        caption_update = {
            "message": {
                "message_id": 22,
                "chat": {"id": chat_id},
                "from": {"id": chat_id},
                "photo": [{"file_id": "cover"}],
                "caption": "https://www.xiaohongshu.com/discovery/item/example",
            }
        }
        await bot.process_admin_update(caption_update)
        fake_link_job.assert_awaited_once()

    async def test_pending_uploaded_video_starts_from_confirm_callback(self):
        chat_id = int(bot.ADMIN_ID)
        self.start_patch(
            patch.object(
                bot,
                "get_user_settings",
                side_effect=lambda _chat: {
                    **bot._with_required_coverage(bot.DEFAULT_SETTINGS),
                    "auto_start": False,
                },
            )
        )
        self.mock_send_text.reset_mock()
        await bot.process_admin_update(
            {
                "message": {
                    "message_id": 31,
                    "chat": {"id": chat_id},
                    "from": {"id": chat_id},
                    "document": {
                        "file_id": "pending-doc-video",
                        "file_name": "episode.mp4",
                        "mime_type": "video/mp4",
                    },
                }
            }
        )

        self.assertEqual(
            bot.PENDING_INPUTS[chat_id]["file_id"], "pending-doc-video"
        )
        confirm_markup = self.mock_send_text.await_args.kwargs["reply_markup"]
        self.assertEqual(confirm_markup, bot.confirm_keyboard())

        fake_video_job = AsyncMock(return_value=None)
        self.start_patch(
            patch.object(bot, "handle_telegram_video_job", new=fake_video_job)
        )
        await bot.process_admin_update(
            {
                "callback_query": {
                    "id": "cb-video-start",
                    "data": "job:start",
                    "from": {"id": chat_id},
                    "message": {"message_id": 32, "chat": {"id": chat_id}},
                }
            }
        )
        # Callback-spawned jobs run in the event loop; give their retained
        # task one loop turn before checking the mocked job.
        await asyncio.sleep(0)
        fake_video_job.assert_awaited_once()
        self.assertEqual(fake_video_job.await_args.args[2]["file_id"], "pending-doc-video")
        self.assertNotIn(chat_id, bot.PENDING_INPUTS)

    async def test_progress_buttons_stay_attached_and_cancel_worker_job(self):
        chat_id = int(bot.ADMIN_ID)
        state = self.create_job(
            job_id="job-button-cancel",
            chat_id=chat_id,
            status_message_id=55,
        )
        tracker = bot.ProgressTracker(chat_id, 55, job_id=state["job_id"])
        self.mock_tg_api.reset_mock()
        await tracker.update("working", force=True)
        self.mock_tg_api.assert_awaited_once()
        method, payload = self.mock_tg_api.await_args.args
        self.assertEqual(method, "editMessageText")
        self.assertEqual(payload["reply_markup"], bot.job_progress_keyboard())

        self.mock_send_text.reset_mock()
        self.mock_edit_text.reset_mock()
        status_callback = {
            "callback_query": {
                "id": "cb-status",
                "data": "menu:status",
                "from": {"id": chat_id},
                "message": {
                    "message_id": 55,
                    "chat": {"id": chat_id},
                },
            }
        }
        await bot.process_admin_update(status_callback)
        # First callback after deploy installs persistent buttons, then the
        # status view is sent separately without overwriting progress.
        self.assertEqual(self.mock_send_text.await_count, 2)
        self.assertTrue(
            self.mock_send_text.await_args_list[0].kwargs["reply_markup"][
                "is_persistent"
            ]
        )
        self.assertEqual(
            self.mock_send_text.await_args_list[1].kwargs["reply_markup"],
            bot.main_menu_keyboard(chat_id),
        )
        self.mock_edit_text.assert_not_awaited()

        callback = {
            "callback_query": {
                "id": "cb-1",
                "data": "menu:cancel",
                "from": {"id": chat_id},
                "message": {
                    "message_id": 55,
                    "chat": {"id": chat_id},
                },
            }
        }
        await bot.process_admin_update(callback)

        loaded = bot._load_worker_job(state["job_id"])
        self.assertIn(state["job_id"], bot.CANCELLED_JOBS)
        self.assertTrue(loaded["cancel_requested"])
        self.assertEqual(loaded["status"], "cancelled")
        self.assertNotIn(chat_id, bot.ACTIVE_JOBS)
        self.mock_edit_text.assert_awaited()

    def test_public_base_url_prefers_actual_render_hostname(self):
        with patch.dict(
            os.environ,
            {
                "RENDER_EXTERNAL_HOSTNAME": "actual-service.onrender.com",
                "RENDER_EXTERNAL_URL": "https://stale-old-service.onrender.com",
            },
            clear=True,
        ):
            self.assertEqual(
                bot._resolve_public_base_url(),
                "https://actual-service.onrender.com",
            )

        with patch.dict(
            os.environ,
            {
                "WEBHOOK_BASE_URL": "https://bot.example.com/",
                "RENDER_EXTERNAL_HOSTNAME": "actual-service.onrender.com",
                "RENDER_EXTERNAL_URL": "https://stale-old-service.onrender.com",
            },
            clear=True,
        ):
            self.assertEqual(
                bot._resolve_public_base_url(), "https://bot.example.com"
            )

        # A stale env copied from an older Render service must never make
        # this freshly deployed instance register Telegram's webhook back
        # to that old service; the platform hostname of *this* instance
        # wins over any different *.onrender.com value.
        for env_key in ("WEBHOOK_BASE_URL", "PUBLIC_BASE_URL"):
            with patch.dict(
                os.environ,
                {
                    env_key: "https://rednote-bot-gzd0.onrender.com/",
                    "RENDER_EXTERNAL_HOSTNAME": "actual-service.onrender.com",
                    "RENDER_EXTERNAL_URL": "https://rednote-bot-gzd0.onrender.com",
                },
                clear=True,
            ):
                self.assertEqual(
                    bot._resolve_public_base_url(),
                    "https://actual-service.onrender.com",
                )

    async def test_telegram_setup_registers_menu_and_callback_webhook(self):
        calls = []
        expected_url = "https://actual-service.onrender.com/webhook"

        async def fake_tg_api(method, data=None, timeout=90.0):
            calls.append((method, data or {}))
            if method == "getMe":
                return {"ok": True, "result": {"username": "recap_test_bot"}}
            if method == "getWebhookInfo":
                return {
                    "ok": True,
                    "result": {
                        "url": expected_url,
                        "allowed_updates": bot.TELEGRAM_WEBHOOK_ALLOWED_UPDATES,
                        "pending_update_count": 0,
                    },
                }
            return {"ok": True, "result": True}

        self.start_patch(patch.object(bot, "tg_api", new=fake_tg_api))

        self.assertTrue(await bot._setup_bot_once(expected_url))
        self.assertEqual(
            [method for method, _ in calls],
            [
                "getMe",
                "setMyCommands",
                "setMyCommands",
                "setChatMenuButton",
                "setWebhook",
                "getWebhookInfo",
            ],
        )
        menu_payload = calls[3][1]
        webhook_payload = calls[4][1]
        self.assertEqual(menu_payload, {"menu_button": {"type": "commands"}})
        self.assertEqual(webhook_payload["url"], expected_url)
        self.assertEqual(
            webhook_payload["allowed_updates"],
            ["message", "edited_message", "callback_query"],
        )

    async def test_background_task_registry_retains_and_cleans_tasks(self):
        finished = asyncio.Event()

        async def quick_job():
            finished.set()

        task = bot._spawn_background(quick_job(), name="test-quick")
        self.assertIn(task, bot.BACKGROUND_TASKS)
        await asyncio.wait_for(finished.wait(), timeout=1)
        await task
        await asyncio.sleep(0)
        self.assertNotIn(task, bot.BACKGROUND_TASKS)

    async def test_lifespan_starts_telegram_setup(self):
        started = asyncio.Event()

        async def fake_setup():
            started.set()
            await asyncio.Event().wait()

        self.start_patch(patch.object(bot, "_setup_bot", new=fake_setup))
        async with bot.lifespan(bot.app):
            await asyncio.wait_for(started.wait(), timeout=1)

    async def test_worker_api_auth_and_health(self):
        async with self.api_client() as client:
            health = await client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["version"], bot.BUILD_VERSION)
            self.assertEqual(
                health.json()["telegram_update_guard"],
                "friendly_retry_on_error",
            )
            self.assertEqual(
                health.json()["manual_part_send_cap_bytes"],
                bot.MANUAL_RECOVERABLE_SEND_BYTES,
            )
            self.assertEqual(
                health.json()["manual_session_recovery"],
                "telegram_message_payload_v1_with_legacy_caption_fallback",
            )
            self.assertEqual(
                health.json()["direct_script_generation"],
                "multi_model_retry_robust_json_empty_scene_recovery",
            )
            self.assertEqual(
                health.json()["direct_script_models"], bot.PREFERRED_MODELS
            )
            self.assertEqual(health.json()["minimum_script_coverage_percent"], 90)
            self.assertEqual(health.json()["target_final_duration_percent"], 90)
            script_rate = health.json()["script_length_rate"]
            self.assertEqual(script_rate["active_provider"], "edge")
            self.assertEqual(script_rate["clamp"], [1.2, 24.0])
            self.assertEqual(script_rate["seed_by_provider"]["edge"], 12.0)
            self.assertEqual(
                script_rate["learning_source"],
                "measured_tts_effective_chars_per_second",
            )
            self.assertAlmostEqual(
                script_rate["by_provider"]["edge"],
                bot.learned_chars_per_second("edge"),
                places=3,
            )
            self.assertEqual(
                health.json()["script_trim_policy"],
                "sentence_then_word_tail_boundaries",
            )
            self.assertEqual(
                health.json()["tts_resilience"],
                "per_scene_retries_then_sentence_pieces_then_fresh_pass",
            )
            self.assertIn("last_tts_retry_recovered_scenes", health.json())
            self.assertIn("last_script_error", health.json())
            self.assertIn("last_script_raw_preview", health.json())
            self.assertIn("last_script_error_by_model", health.json())
            self.assertEqual(health.json()["last_script_recovery"], "")
            self.assertIn("last_duration_calibration", health.json())
            self.assertEqual(
                health.json()["job_registry"],
                "disk_mirrored_active_jobs_with_stall_watchdog",
            )
            self.assertIn("active_jobs", health.json())
            self.assertIn("last_job_results", health.json())
            self.assertIn("interrupted_jobs_last_restart", health.json())
            self.assertGreaterEqual(
                health.json()["job_watchdog"]["stall_seconds"], 120
            )
            self.assertEqual(
                health.json()["duration_calibration"],
                "script_first_dedupe_expand_short_shave_over_source",
            )
            self.assertEqual(
                health.json()["script_plain_text_fallback"],
                "plain_text_retelling_split_locally_then_prose_salvage",
            )
            self.assertEqual(health.json()["last_tts_silenced_scenes"], 0)
            self.assertEqual(
                health.json()["final_duration_enforcement"],
                "measured_tts_and_rendered_file",
            )
            self.assertEqual(
                health.json()["first_person_enforcement"], "per_scene_validated"
            )
            self.assertEqual(health.json()["source_segment_seconds"], 120)
            self.assertEqual(health.json()["recap_flow"], "manual_sequential_per_part")
            self.assertTrue(health.json()["manual_parts_enabled"])
            self.assertEqual(
                health.json()["segment_pipeline"],
                "download_plan_120s_extract_one_send_click_recap_then_next_explicit_join",
            )
            self.assertEqual(
                health.json()["input_types"],
                ["telegram_video", "telegram_document_video", "video_link"],
            )
            self.assertEqual(health.json()["telegram_file_download_limit_mb"], 20)
            self.assertEqual(
                health.json()["telegram_link_handling"],
                "profile_rejected_public_post_web_fallback",
            )
            self.assertEqual(
                health.json()["source_download_diagnostics"],
                "redacted_yt_dlp_stderr_with_direct_rednote_and_telegram_fallbacks",
            )
            self.assertEqual(
                health.json()["telegram_buttons"],
                "inline_persistent_menu_auto_one_shot_manual_opt_in",
            )
            self.assertEqual(
                health.json()["telegram_pipeline"], "direct_manual_opt_in"
            )
            self.assertFalse(health.json()["n8n_telegram_handoff_enabled"])
            self.assertEqual(
                health.json()["auto_retry"]["segment_attempts"],
                bot.AUTO_SEGMENT_ATTEMPTS,
            )
            self.assertEqual(
                health.json()["auto_retry"]["source_attempts"],
                bot.AUTO_SOURCE_ATTEMPTS,
            )
            self.assertEqual(
                health.json()["auto_retry"]["partial_final_delivery"],
                "rejected_as_failed",
            )
            self.assertEqual(
                health.json()["source_split_acceptance"],
                "stream_copy_keyframe_snapped_max_1p5x_window_then_720p_ultrafast_fallback",
            )
            self.assertEqual(
                health.json()["render_keepalive"],
                "self_ping_public_healthz_while_jobs_active",
            )
            self.assertEqual(
                health.json()["gemini_generate_timeout_seconds"],
                bot.GEMINI_GENERATE_TIMEOUT_SECONDS,
            )
            self.assertEqual(
                health.json()["gemini_tts_timeout_seconds"],
                bot.GEMINI_TTS_TIMEOUT_SECONDS,
            )
            self.assertEqual(health.json()["default_voice"], "Charon")
            self.assertFalse(health.json()["default_deep_voice_filter"])
            self.assertEqual(
                health.json()["gemini_overload_retry_budget_seconds"],
                bot.GEMINI_OVERLOAD_RETRY_BUDGET_SECONDS,
            )
            self.assertEqual(
                health.json()["gemini_overload_max_rounds"],
                bot.GEMINI_OVERLOAD_MAX_ROUNDS,
            )
            self.assertIn(
                "gemini-3.5-flash-lite", health.json()["direct_script_models"]
            )
            self.assertNotIn(
                "gemini-2.5-flash-lite", health.json()["direct_script_models"]
            )
            self.assertEqual(
                health.json()["gemini_hedge_delay_seconds"],
                bot.GEMINI_HEDGE_DELAY_SECONDS,
            )
            self.assertEqual(
                health.json()["gemini_hedge_initial_parallel"],
                bot.GEMINI_HEDGE_INITIAL_PARALLEL,
            )
            self.assertEqual(
                health.json()["unavailable_gemini_models"],
                sorted(bot._DEAD_GEMINI_MODELS),
            )
            self.assertEqual(
                health.json()["direct_tts_models"], bot.PREFERRED_TTS_MODELS
            )
            self.assertIn(
                "gemini-3.8-flash-lite-tts", health.json()["direct_tts_models"]
            )
            self.assertEqual(
                health.json()["last_tts_error"], bot._LAST_TTS_ERROR
            )
            self.assertEqual(
                health.json()["last_tts_error_by_model"],
                dict(bot._LAST_TTS_ERROR_BY_MODEL),
            )
            self.assertEqual(
                health.json()["quota_exhausted_models"],
                bot._quota_exhausted_active(),
            )
            self.assertIn(
                "gemini-2.5-flash-tts", health.json()["direct_tts_models"]
            )
            self.assertEqual(
                health.json()["last_tts_provider"], bot._LAST_TTS_PROVIDER
            )
            self.assertTrue(health.json()["edge_tts_fallback_enabled"])
            self.assertEqual(
                health.json()["tts_provider_order"],
                list(bot.TTS_PROVIDER_ORDER),
            )
            self.assertEqual(health.json()["tts_provider_order"][0], "edge")
            self.assertEqual(
                health.json()["telegram_webhook_allowed_updates"],
                bot.TELEGRAM_WEBHOOK_ALLOWED_UPDATES,
            )
            self.assertEqual(health.json()["public_base_url"], bot.PUBLIC_BASE_URL)
            self.assertEqual(health.json()["render_git_commit"], bot.RENDER_GIT_COMMIT)
            self.assertEqual(health.json()["render_git_branch"], bot.RENDER_GIT_BRANCH)
            self.assertEqual(
                health.json()["render_git_repo_slug"], bot.RENDER_GIT_REPO_SLUG
            )

            legacy_health = await client.get("/healthz")
            self.assertEqual(legacy_health.status_code, 200)
            self.assertEqual(legacy_health.json(), health.json())

            root = await client.get("/")
            self.assertEqual(root.status_code, 200)
            self.assertEqual(
                root.json()["direct_script_generation"],
                "multi_model_retry_robust_json_empty_scene_recovery",
            )
            self.assertEqual(root.json()["minimum_script_coverage_percent"], 90)
            self.assertEqual(root.json()["target_final_duration_percent"], 90)
            self.assertEqual(root.json()["source_segment_seconds"], 120)
            self.assertEqual(root.json()["recap_flow"], "manual_sequential_per_part")
            self.assertEqual(
                root.json()["segment_pipeline"],
                "download_plan_120s_extract_one_send_click_recap_then_next_explicit_join",
            )
            self.assertTrue(root.json()["worker_api_configured"])

            missing_auth = await client.get("/api/v1/jobs/nope")
            self.assertEqual(missing_auth.status_code, 401)

            wrong_auth = await client.get(
                "/api/v1/jobs/nope", headers={"X-Worker-Key": "wrong"}
            )
            self.assertEqual(wrong_auth.status_code, 401)

            authorized_missing = await client.get(
                "/api/v1/jobs/nope",
                headers={"Authorization": "Bearer test-worker-key"},
            )
            self.assertEqual(authorized_missing.status_code, 404)

            with patch.object(bot, "WORKER_API_KEY", ""):
                unconfigured = await client.get(
                    "/api/v1/jobs/nope", headers=self.auth_headers
                )
            self.assertEqual(unconfigured.status_code, 503)

    async def test_worker_api_create_is_idempotent_and_conflict_checked(self):
        payload = {
            "job_id": "job-api-1",
            "chat_id": 11001,
            "url": "https://videos.example/a",
            "reply_msg_id": 10,
            "create_status_message": False,
            "settings": {"voice": "Kore", "pov": "third", "unknown": "ignored"},
        }
        async with self.api_client() as client:
            created = await client.post(
                "/api/v1/jobs", json=payload, headers=self.auth_headers
            )
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["job_id"], "job-api-1")
            self.assertEqual(created.json()["selected_settings"]["voice"], "Kore")
            state = bot._load_worker_job("job-api-1")
            self.assertEqual(state["settings"]["pov"], "first")

            duplicate = await client.post(
                "/api/v1/jobs",
                json={**payload, "status_message_id": 88},
                headers=self.auth_headers,
            )
            self.assertEqual(duplicate.status_code, 200)
            self.assertEqual(duplicate.json()["status_message_id"], 88)

            different_url = await client.post(
                "/api/v1/jobs",
                json={**payload, "url": "https://videos.example/different"},
                headers=self.auth_headers,
            )
            self.assertEqual(different_url.status_code, 409)

            another_active = await client.post(
                "/api/v1/jobs",
                json={**payload, "job_id": "job-api-2"},
                headers=self.auth_headers,
            )
            self.assertEqual(another_active.status_code, 409)

            nested = await client.post(
                "/api/v1/jobs",
                json={
                    "body": {
                        **payload,
                        "job_id": "job-api-nested",
                        "chat_id": 11002,
                    }
                },
                headers=self.auth_headers,
            )
            self.assertEqual(nested.status_code, 200)
            self.assertEqual(nested.json()["job_id"], "job-api-nested")

    async def test_worker_api_stage_prerequisites_idempotency_and_done(self):
        state = self.create_job(job_id="job-api-stage", chat_id=12001)
        started, release, calls = self.install_fake_stage(wait=True)

        async with self.api_client() as client:
            prerequisite = await client.post(
                "/api/v1/jobs/job-api-stage/stages/transcribe",
                headers=self.auth_headers,
            )
            self.assertEqual(prerequisite.status_code, 409)

            first = await client.post(
                "/api/v1/jobs/job-api-stage/stages/download",
                headers=self.auth_headers,
            )
            self.assertEqual(first.status_code, 202)
            await asyncio.wait_for(started.wait(), timeout=2)
            task = bot.WORKER_TASKS[state["job_id"]]

            duplicate = await client.post(
                "/api/v1/jobs/job-api-stage/stages/download",
                headers=self.auth_headers,
            )
            self.assertEqual(duplicate.status_code, 202)

            conflict = await client.post(
                "/api/v1/jobs/job-api-stage/stages/transcribe",
                headers=self.auth_headers,
            )
            self.assertEqual(conflict.status_code, 409)

            release.set()
            await asyncio.wait_for(task, timeout=2)
            done_duplicate = await client.post(
                "/api/v1/jobs/job-api-stage/stages/download",
                headers=self.auth_headers,
            )
            self.assertEqual(done_duplicate.status_code, 200)
            self.assertEqual(calls, ["job-api-stage"])

        completed = bot._load_worker_job("job-api-stage")
        self.assertEqual(completed["stages"]["download"]["status"], "done")
        self.assertEqual(completed["status"], "stage_done")

    async def test_worker_api_failed_stage_can_be_restarted(self):
        state = self.create_job(job_id="job-api-restart", chat_id=12002)
        state = bot._load_worker_job(state["job_id"])
        state["status"] = "failed"
        state["current_stage"] = "download"
        state["terminal_stat_counted"] = True
        bot._set_worker_stage(state, "download", "failed", error="boom")
        bot._save_worker_job(state)
        _, _, calls = self.install_fake_stage(wait=False)

        async with self.api_client() as client:
            response = await client.post(
                "/api/v1/jobs/job-api-restart/stages/download",
                headers=self.auth_headers,
            )
            self.assertEqual(response.status_code, 202)
            task = bot.WORKER_TASKS["job-api-restart"]
            await asyncio.wait_for(task, timeout=2)

        completed = await self.wait_for_stage_status(
            "job-api-restart", "download", "done"
        )
        self.assertEqual(calls, ["job-api-restart"])
        self.assertEqual(completed["status"], "stage_done")
        self.assertFalse(completed["terminal_stat_counted"])

    async def test_worker_api_cancel_idle_fail_and_cleanup(self):
        self.create_job(job_id="job-api-cancel", chat_id=13001)
        self.create_job(job_id="job-api-fail", chat_id=13002)

        async with self.api_client() as client:
            cancelled = await client.post(
                "/api/v1/jobs/job-api-cancel/cancel", headers=self.auth_headers
            )
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(cancelled.json()["status"], "cancelled")
            cancelled_state = bot._load_worker_job("job-api-cancel")
            self.assertEqual(
                cancelled_state["stages"]["download"]["status"], "cancelled"
            )
            self.assertNotIn(13001, bot.ACTIVE_JOBS)

            failed = await client.post(
                "/api/v1/jobs/job-api-fail/fail",
                json={"reason": "workflow stopped"},
                headers=self.auth_headers,
            )
            self.assertEqual(failed.status_code, 200)
            self.assertEqual(failed.json()["status"], "failed")
            self.assertEqual(failed.json()["error"], "workflow stopped")
            self.assertNotIn(13002, bot.ACTIVE_JOBS)

            deleted = await client.delete(
                "/api/v1/jobs/job-api-cancel", headers=self.auth_headers
            )
            self.assertEqual(deleted.status_code, 200)
            self.assertTrue(deleted.json()["deleted"])
            self.assertFalse((self.jobs_dir / "job-api-cancel").exists())

            deleted_again = await client.delete(
                "/api/v1/jobs/job-api-cancel", headers=self.auth_headers
            )
            self.assertEqual(deleted_again.status_code, 200)
            self.assertFalse(deleted_again.json()["deleted"])

    async def test_worker_api_cancel_running_stage(self):
        self.create_job(job_id="job-api-cancel-running", chat_id=13003)
        started, _, _ = self.install_fake_stage(wait=True)

        async with self.api_client() as client:
            started_response = await client.post(
                "/api/v1/jobs/job-api-cancel-running/stages/download",
                headers=self.auth_headers,
            )
            self.assertEqual(started_response.status_code, 202)
            await asyncio.wait_for(started.wait(), timeout=2)
            task = bot.WORKER_TASKS["job-api-cancel-running"]

            cancel_response = await client.post(
                "/api/v1/jobs/job-api-cancel-running/cancel",
                headers=self.auth_headers,
            )
            self.assertEqual(cancel_response.status_code, 202)
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)

            status = await client.get(
                "/api/v1/jobs/job-api-cancel-running", headers=self.auth_headers
            )
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["status"], "cancelled")
            self.assertFalse(status.json()["task_running"])

        state = bot._load_worker_job("job-api-cancel-running")
        self.assertEqual(state["stages"]["download"]["status"], "cancelled")
        self.assertNotIn(13003, bot.ACTIVE_JOBS)

    async def test_telegram_source_is_recapped_part_by_part_then_joined_and_sent(self):
        # This is the default one-shot auto path (v5): one send, internal
        # 2-minute windows, automatic join and final delivery.  The legacy
        # manual button flow (explicit opt-in) has dedicated tests below.
        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", False))
        build_calls = []
        prepare_calls = []
        concat_calls = []

        async def fake_telegram_download(file_id, output_path, tracker=None, **_kwargs):
            self.assertEqual(file_id, "uploaded-source")
            Path(output_path).write_bytes(b"x" * 10001)
            return 250.0

        async def fake_prepare(source_path, work_dir, total_duration, segment_seconds, tracker=None, **_kwargs):
            prepare_calls.append((source_path, work_dir, total_duration, segment_seconds))
            segment_dir = Path(work_dir) / "source_segments"
            segment_dir.mkdir(parents=True, exist_ok=True)
            segments = []
            for index, start, end in ((1, 0, 120), (2, 120, 240), (3, 240, 250)):
                path = segment_dir / f"source_part_{index:04d}.mp4"
                path.write_bytes(b"s" * 100)
                segments.append(
                    {
                        "index": index,
                        "path": str(path),
                        "start": float(start),
                        "end": float(end),
                        "duration": float(end - start),
                    }
                )
            return segments

        async def fake_build(
            chunk_path,
            actual_dur,
            start,
            end,
            story_context,
            part_num,
            tracker,
            settings,
            **kwargs,
        ):
            kind = "script" if kwargs.get("script_only") else "render"
            build_calls.append(
                {
                    "kind": kind,
                    "path": chunk_path,
                    "duration": actual_dur,
                    "start": start,
                    "end": end,
                    "context": dict(story_context),
                    "part": part_num,
                    "settings": dict(settings),
                }
            )
            summary = {
                "characters": ["Hero", "Mina"],
                "summary": f"after part {part_num}",
                "last_scene": f"scene ending {part_num}",
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
            }
            if kind == "script":
                return {
                    "version": 1,
                    "part_index": part_num,
                    "scenes": [
                        {
                            "start": 0.0,
                            "end": 10.0,
                            "burmese": f"{MALE_PRONOUN} အပိုင်း {part_num} အကြောင်း ပြောပြတယ်။",
                        }
                    ],
                    "cuts": [],
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                    "story_context_before": dict(story_context),
                    "story_context_after": summary,
                    "metrics": {"target_chars": 100},
                }, summary
            output = Path(chunk_path).parents[1] / f"recap_part_{part_num:04d}.mp4"
            output.write_bytes(b"r" * 100)
            return str(output), summary

        def fake_concat(parts, output_path, concat_path):
            concat_calls.append(list(parts))
            Path(concat_path).write_text("\n".join(parts), encoding="utf-8")
            Path(output_path).write_bytes(b"joined" * 100)
            return True

        def fake_probe(path):
            name = Path(path).name
            if name == "final_recap.mp4":
                return (225.0, 64, 64)
            if name.endswith("0003.mp4"):
                return (10.0, 64, 64)
            return (120.0, 64, 64)

        self.start_patch(
            patch.object(bot, "download_telegram_file", new=fake_telegram_download)
        )
        self.start_patch(
            patch.object(bot, "_prepare_source_segments", new=fake_prepare)
        )
        self.start_patch(patch.object(bot, "build_chunk_recap", new=fake_build))
        self.start_patch(patch.object(bot, "_concat_video_parts", new=fake_concat))
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))
        self.start_patch(
            patch.object(
                bot,
                "split_for_telegram",
                new=AsyncMock(side_effect=lambda path, _work_dir: [path]),
            )
        )
        self.mock_tg_upload.reset_mock()

        await bot.handle_telegram_video_job(
            15001,
            88,
            {
                "file_id": "uploaded-source",
                "file_name": "movie.mp4",
                "mime_type": "video/mp4",
                "duration": 250,
                "file_size": 10001,
            },
        )

        self.assertEqual(len(prepare_calls), 1)
        self.assertEqual(prepare_calls[0][2:], (250.0, 120))
        # Script-first (user's order): *every* window is scripted
        # before the first window is voiced/edited.
        script_calls = [c for c in build_calls if c["kind"] == "script"]
        render_calls = [c for c in build_calls if c["kind"] == "render"]
        self.assertEqual([call["part"] for call in script_calls], [1, 2, 3])
        self.assertEqual([call["part"] for call in render_calls], [1, 2, 3])
        last_script_index = max(
            i for i, c in enumerate(build_calls) if c["kind"] == "script"
        )
        first_render_index = min(
            i for i, c in enumerate(build_calls) if c["kind"] == "render"
        )
        self.assertLess(last_script_index, first_render_index)
        self.assertEqual(
            [(call["start"], call["end"]) for call in script_calls],
            [(0.0, 120.0), (120.0, 240.0), (240.0, 250.0)],
        )
        self.assertTrue(
            all(call["settings"]["chunk"] == 120 for call in build_calls)
        )
        self.assertEqual(script_calls[1]["context"]["summary"], "after part 1")
        self.assertEqual(script_calls[2]["context"]["summary"], "after part 2")
        self.assertEqual(script_calls[2]["context"]["protagonist"], "Hero")
        self.assertEqual(script_calls[2]["context"]["characters"], ["Hero", "Mina"])
        self.assertEqual(len(concat_calls), 1)
        self.assertEqual(
            [Path(path).name for path in concat_calls[0]],
            ["recap_part_0001.mp4", "recap_part_0002.mp4", "recap_part_0003.mp4"],
        )
        self.mock_tg_upload.assert_awaited_once()
        self.assertEqual(self.mock_tg_upload.await_args.args[0], "sendVideo")
        self.assertEqual(Path(self.mock_tg_upload.await_args.args[4]).name, "final_recap.mp4")
        self.assertNotIn(15001, bot.ACTIVE_JOBS)

    async def test_failed_final_join_never_sends_a_partial_video(self):
        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", False))

        async def fake_telegram_download(_file_id, output_path, _tracker=None, **_kwargs):
            Path(output_path).write_bytes(b"x" * 10001)
            return 60.0

        async def fake_prepare(_source_path, work_dir, _duration, _seconds, tracker=None, **_kwargs):
            path = Path(work_dir) / "source_part_0001.mp4"
            path.write_bytes(b"s" * 100)
            return [
                {
                    "index": 1,
                    "path": str(path),
                    "start": 0.0,
                    "end": 60.0,
                    "duration": 60.0,
                }
            ]

        async def fake_build(chunk_path, *_args, **_kwargs):
            summary = {
                "characters": ["Hero"],
                "summary": "done",
                "last_scene": "end",
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
            }
            if _kwargs.get("script_only"):
                return {
                    "scenes": [
                        {
                            "start": 0.0,
                            "end": 10.0,
                            "burmese": f"{MALE_PRONOUN} ပြောပြတယ်။",
                        }
                    ],
                    "cuts": [],
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                    "story_context_after": summary,
                    "metrics": {"target_chars": 100},
                }, summary
            output = Path(chunk_path).parent / "recap_part_0001.mp4"
            output.write_bytes(b"r" * 100)
            return str(output), summary

        self.start_patch(
            patch.object(bot, "download_telegram_file", new=fake_telegram_download)
        )
        self.start_patch(
            patch.object(bot, "_prepare_source_segments", new=fake_prepare)
        )
        self.start_patch(patch.object(bot, "build_chunk_recap", new=fake_build))
        self.start_patch(
            patch.object(bot, "_concat_video_parts", new=Mock(return_value=False))
        )
        self.start_patch(
            patch.object(bot, "probe_video_info", new=lambda _path: (60.0, 64, 64))
        )
        self.mock_tg_upload.reset_mock()
        self.mock_edit_text.reset_mock()

        await bot.handle_telegram_video_job(
            15002,
            89,
            {"file_id": "uploaded-source", "mime_type": "video/mp4"},
        )

        self.mock_tg_upload.assert_not_awaited()
        self.assertTrue(
            any(
                "final video တစ်ခုအဖြစ်" in str(call.args[2])
                for call in self.mock_edit_text.await_args_list
            )
        )
        self.assertNotIn(15002, bot.ACTIVE_JOBS)

    def test_source_url_extraction_rednote_candidates_and_secret_redaction(self):
        shared = "ဒါလေးကြည့် https://xhslink.com/a/AbC123, copy လုပ်ထား"
        self.assertEqual(
            bot._extract_first_url(shared), "https://xhslink.com/a/AbC123"
        )
        self.assertTrue(bot._is_rednote_url("https://xhslink.cn/a/test"))
        self.assertTrue(
            bot._is_rednote_url("https://www.xiaohongshu.com/explore/test")
        )

        page = (
            r'{"masterUrl":"https:\/\/sns-video-bd.xhscdn.com\/movie.mp4?sign=1",'
            r'"originVideoKey":"origin-key-1"}'
        )
        candidates = bot._extract_direct_video_urls_from_text(
            page, "https://www.xiaohongshu.com/explore/test"
        )
        self.assertEqual(
            candidates[0], "https://sns-video-bd.xhscdn.com/movie.mp4?sign=1"
        )
        self.assertIn(
            "https://sns-video-bd.xhscdn.com/origin-key-1", candidates
        )

        fake_token = "123456" + ":" + ("A" * 35)
        self.start_patch(patch.object(bot, "BOT_TOKEN", fake_token))
        redacted = bot._redact_secret_text(
            f"GET https://api.telegram.org/bot{fake_token}/getMe token={fake_token}"
        )
        self.assertNotIn(fake_token, redacted)
        self.assertIn("[REDACTED]", redacted)
        self.assertEqual(bot.logging.getLogger("httpx").level, bot.logging.WARNING)
        self.assertEqual(
            bot.logging.getLogger("httpcore").level, bot.logging.WARNING
        )
        mapped = bot._download_error_from_detail(
            f"ERROR: HTTP Error 403: Forbidden using {fake_token}"
        )
        self.assertIn("403/anti-bot", mapped.user_message)
        self.assertNotIn(fake_token, mapped.detail)

    async def test_telegram_profile_link_is_rejected_before_extractor(self):
        profile_url = "https://t.me/Goiiautobot0"
        public_post_url = "https://t.me/example/123"
        self.assertEqual(bot._telegram_link_info(profile_url)["kind"], "profile")
        self.assertEqual(
            bot._telegram_link_info(public_post_url)["kind"], "public_post"
        )
        self.assertEqual(
            bot._telegram_link_info("https://t.me/s/example/123")["kind"],
            "public_post",
        )
        self.assertEqual(
            bot._telegram_link_info("https://t.me/c/123456/789")["kind"],
            "private_post",
        )
        self.assertEqual(
            bot._telegram_link_info("https://t.me/+invite-token")["kind"],
            "invite",
        )

        mapped = bot._download_error_from_detail(
            f"ERROR: Unsupported URL: {profile_url}", profile_url
        )
        self.assertIn("profile link", mapped.user_message)
        self.assertIn("forward", mapped.user_message)
        self.assertIn("extractor rejected", mapped.detail)
        self.assertNotIn(profile_url, mapped.detail)

        info_mock = AsyncMock(
            side_effect=AssertionError("Telegram links must skip yt-dlp info")
        )
        self.start_patch(patch.object(bot, "_run_ytdlp_info", new=info_mock))
        self.assertEqual(await bot.get_source_duration(profile_url), 0.0)
        with self.assertRaises(bot.SourceDownloadError) as raised_duration:
            await bot.get_source_duration(profile_url, raise_on_error=True)
        self.assertIn("profile link", raised_duration.exception.user_message)
        info_mock.assert_not_awaited()

        ytdlp_mock = AsyncMock(
            side_effect=AssertionError("profile links must not reach yt-dlp")
        )
        self.start_patch(
            patch.object(bot, "_download_with_ytdlp_once", new=ytdlp_mock)
        )
        with self.assertRaises(bot.SourceDownloadError) as raised_download:
            await bot.download_full(
                profile_url,
                str(self.tmp_path / "telegram-profile.mp4"),
                raise_on_error=True,
            )
        self.assertIn("profile link", raised_download.exception.user_message)
        self.assertIn("https://t.me/name/123", raised_download.exception.user_message)
        ytdlp_mock.assert_not_awaited()

        page = (
            '<img src="https://cdn4.telesco.pe/file/poster.jpg">'
            '<video preload="none" '
            'src="https://cdn4.telesco.pe/file/movie.mp4"></video>'
        )
        self.assertEqual(
            bot._extract_telegram_video_urls_from_html(page, public_post_url),
            ["https://cdn4.telesco.pe/file/movie.mp4"],
        )

        extract_mock = AsyncMock(
            return_value=["https://cdn4.telesco.pe/file/movie.mp4"]
        )
        direct_mock = AsyncMock(return_value=33.0)
        self.start_patch(
            patch.object(bot, "_extract_telegram_direct_urls", new=extract_mock)
        )
        self.start_patch(
            patch.object(bot, "_download_direct_media", new=direct_mock)
        )
        duration = await bot.download_full(
            public_post_url,
            str(self.tmp_path / "telegram-public.mp4"),
            raise_on_error=True,
        )
        self.assertEqual(duration, 33.0)
        extract_mock.assert_awaited_once_with(public_post_url)
        direct_mock.assert_awaited_once()
        ytdlp_mock.assert_not_awaited()

    def test_rednote_cookies_accept_netscape_and_json(self):
        self.start_patch(
            patch.object(bot, "_configured_cookie_file_path", return_value="")
        )
        netscape = "xhslink.com\tFALSE\t/\tFALSE\t0\ta1\tcookie-value"
        self.start_patch(patch.object(bot, "XHS_COOKIE", netscape))
        self.assertEqual(
            bot._cookie_records_from_value(netscape),
            [("a1", "cookie-value", ".xiaohongshu.com", "/")],
        )
        generated = Path(bot._yt_dlp_cookies_file())
        self.assertTrue(generated.is_file())
        self.assertIn(
            ".xiaohongshu.com\tTRUE\t/\tFALSE\t0\ta1\tcookie-value",
            generated.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            bot._cookie_records_from_value(
                '[{"name":"web_session","value":"abc","domain":".xiaohongshu.com"}]'
            ),
            [("web_session", "abc", ".xiaohongshu.com", "/")],
        )
        self.assertTrue(
            bot._is_direct_media_url(
                "https://sns-video-bd.xhscdn.com/origin-key-without-extension"
            )
        )

    async def test_download_telegram_file_over_20mb_gives_clear_error(self):
        with self.assertRaises(bot.SourceDownloadError) as raised:
            await bot.download_telegram_file(
                "too-big-file",
                str(self.tmp_path / "too-big.mp4"),
                expected_size=21 * 1024 * 1024,
                raise_on_error=True,
            )
        self.assertIn("20MB", raised.exception.user_message)
        self.assertIn("link", raised.exception.user_message)

    async def test_ytdlp_stderr_is_captured_instead_of_hidden(self):
        class FakeStream:
            def __init__(self, payload=b""):
                self.payload = payload
                self.sent = False

            async def read(self, _size):
                if self.sent:
                    return b""
                self.sent = True
                return self.payload

        class FakeProc:
            pid = 999999
            returncode = None

            def __init__(self):
                self.stdout = FakeStream()
                self.stderr = FakeStream(b"ERROR: Unsupported URL: example\n")

            async def wait(self):
                self.returncode = 1
                return 1

        self.start_patch(
            patch.object(
                bot.asyncio,
                "create_subprocess_exec",
                new=AsyncMock(return_value=FakeProc()),
            )
        )
        with self.assertRaises(bot.SourceDownloadError) as raised:
            await bot._download_with_ytdlp_once(
                "https://videos.example/unsupported",
                str(self.tmp_path / "source.mp4"),
            )
        self.assertIn("Unsupported URL", raised.exception.detail)
        self.assertIn("extractor", raised.exception.user_message)

    async def test_direct_media_download_streams_file(self):
        self.start_patch(
            patch.object(bot, "probe_video_info", new=Mock(return_value=(12.0, 640, 360)))
        )

        class FakeResponse:
            status_code = 200
            headers = {"content-type": "video/mp4", "content-length": "11000"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def aiter_bytes(self, _size):
                yield b"0123456789" * 1100

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def stream(self, _method, _url):
                return FakeResponse()

        self.start_patch(patch.object(bot.httpx, "AsyncClient", FakeClient))
        output = self.tmp_path / "direct.mp4"
        duration = await bot._download_direct_media(
            "https://cdn.example/movie.mp4", str(output)
        )
        self.assertEqual(duration, 12.0)
        self.assertEqual(output.read_bytes(), b"0123456789" * 1100)

    async def test_generate_chunk_script_returns_script_without_tts(self):
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/script-only",
        )
        payload = {
            "protagonist": "Hero",
            "pronoun": MALE_PRONOUN,
            "scenes": [
                {"start": 0, "end": 3, "burmese": first_person_text(27)}
            ],
            "cuts": [],
        }
        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(
                generate_content=lambda model, contents, config: SimpleNamespace(
                    text=json.dumps(payload, ensure_ascii=False)
                )
            ),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "extract_story_summary",
                new=AsyncMock(
                    return_value={
                        "characters": ["Hero"],
                        "summary": "opening",
                        "last_scene": "door",
                        "protagonist": "Hero",
                        "pronoun": MALE_PRONOUN,
                    }
                ),
            )
        )
        fake_tts = AsyncMock(side_effect=AssertionError("TTS must not run"))
        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_tts))
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        script = await bot.generate_chunk_script(
            str(self.tmp_path / "chunk.mp4"),
            3,
            0,
            3,
            {},
            1,
            tracker,
            settings={"coverage": 90, "rate": 10.0, "pause": 0},
        )

        self.assertEqual(script["protagonist"], "Hero")
        self.assertEqual(script["pronoun"], MALE_PRONOUN)
        self.assertEqual(script["story_context_after"]["summary"], "opening")
        self.assertEqual(script["metrics"]["actual_chars"], 27)
        fake_tts.assert_not_awaited()

    async def test_generate_chunk_script_falls_back_to_next_model_and_alias_json(self):
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/script-fallback",
        )
        payload = {
            "main_character": {"name": "Hero"},
            "first_person_pronoun": MALE_PRONOUN,
            "segments": [
                {
                    "start_time": "0",
                    "end_time": "3",
                    "text": first_person_text(27),
                }
            ],
            "cuts": [],
        }
        model_calls = []

        def fake_generate(model, contents, config):
            model_calls.append(model)
            if model == bot.PREFERRED_MODELS[0]:
                return SimpleNamespace(
                    text="",
                    candidates=[
                        SimpleNamespace(
                            finish_reason="MAX_TOKENS",
                            safety_ratings=[],
                            content=SimpleNamespace(parts=[]),
                        )
                    ],
                    prompt_feedback=SimpleNamespace(block_reason="OTHER"),
                    usage_metadata=SimpleNamespace(
                        candidates_token_count=0,
                        total_token_count=123,
                    ),
                )
            return SimpleNamespace(
                text="```json\n"
                + json.dumps(payload, ensure_ascii=False)
                + "\n```"
            )

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=fake_generate),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "extract_story_summary",
                new=AsyncMock(
                    return_value={
                        "characters": ["Hero"],
                        "summary": "opening",
                        "last_scene": "door",
                        "protagonist": "Hero",
                        "pronoun": MALE_PRONOUN,
                    }
                ),
            )
        )
        fake_tts = AsyncMock(side_effect=AssertionError("TTS must not run"))
        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_tts))
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        script = await bot.generate_chunk_script(
            str(self.tmp_path / "chunk.mp4"),
            3,
            0,
            3,
            {},
            1,
            tracker,
            settings={"coverage": 90, "rate": 10.0, "pause": 0},
        )

        self.assertEqual(model_calls[:2], bot.PREFERRED_MODELS[:2])
        self.assertEqual(script["protagonist"], "Hero")
        self.assertEqual(script["metrics"]["actual_chars"], 27)
        fake_tts.assert_not_awaited()

    async def test_generate_chunk_script_retries_full_script_when_scenes_empty(self):
        # Counts prompts to verify the retry flow; pin racing off
        # (covered separately) for a deterministic call mapping.
        self.start_patch(
            patch.object(bot, "GEMINI_HEDGE_INITIAL_PARALLEL", 1)
        )
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/script-empty-retry",
        )
        prompts = []

        def fake_generate(model, contents, config):
            prompt = str(contents[1]) if len(contents) > 1 else ""
            prompts.append(prompt)
            if "သုံးနိုင်သော scene မရခဲ့ပါ" in prompt:
                payload = {
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                    "scenes": [
                        {
                            "start": 0,
                            "end": 3,
                            "burmese": first_person_text(27),
                        }
                    ],
                    "cuts": [],
                }
            else:
                payload = {"protagonist": "", "pronoun": "", "scenes": []}
            return SimpleNamespace(
                text=json.dumps(payload, ensure_ascii=False)
            )

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=fake_generate),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "extract_story_summary",
                new=AsyncMock(
                    return_value={
                        "characters": ["Hero"],
                        "summary": "opening",
                        "last_scene": "door",
                        "protagonist": "Hero",
                        "pronoun": MALE_PRONOUN,
                    }
                ),
            )
        )
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        script = await bot.generate_chunk_script(
            str(self.tmp_path / "chunk.mp4"),
            3,
            0,
            3,
            {},
            1,
            tracker,
            settings={"coverage": 90, "rate": 10.0, "pause": 0},
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn('"scenes" ကို ဘယ်တော့မှ ကွက်လပ်မထားရ', prompts[1])
        self.assertEqual(script["metrics"]["actual_chars"], 27)

    async def test_generate_chunk_script_all_model_failures_report_no_scenes(self):
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/script-all-fail",
        )
        model_calls = []

        def fake_generate(model, contents, config):
            model_calls.append(model)
            raise RuntimeError("synthetic empty Gemini response")

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=fake_generate),
        )
        self.start_patch(
            patch.object(bot, "PREFERRED_MODELS", ["model-a", "model-b"])
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        fake_tts = AsyncMock(side_effect=AssertionError("TTS must not run"))
        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_tts))
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        with self.assertRaises(RuntimeError) as raised:
            await bot.generate_chunk_script(
                str(self.tmp_path / "chunk.mp4"),
                120,
                0,
                120,
                {},
                1,
                tracker,
                settings={"coverage": 90, "rate": 6.5, "pause": 0},
            )

        # The initial call itself failed on every model, so the retry
        # prompt is skipped (it would repeat the identical wait);
        # only the initial pass over the models ran.
        self.assertEqual(len(model_calls), 2)
        self.assertIn("Script scene ထုတ်မပေးနိုင်ပါ", str(raised.exception))
        self.assertIn("synthetic empty Gemini response", str(raised.exception))
        self.assertNotIn("0/702 chars", str(raised.exception))
        fake_tts.assert_not_awaited()

    async def test_manual_flow_sends_only_first_part_without_scripting_rest(self):
        events = []
        extract_calls = []

        async def fake_telegram_download(_file_id, output_path, _tracker=None, **_kwargs):
            Path(output_path).write_bytes(b"x" * 10001)
            return 250.0

        def fake_extract(_source_path, output_path, start, duration):
            index = int(Path(output_path).stem.rsplit("_", 1)[-1])
            extract_calls.append((index, float(start), float(duration)))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"s" * 100)
            return True

        async def fake_upload(method, payload, _field, file_name, file_path, _mime):
            events.append(("upload", file_name))
            return {
                "ok": True,
                "result": {"message_id": 1000 + len(events)},
            }

        def fake_probe(path):
            if Path(path).name.endswith("0003.mp4"):
                return (10.0, 64, 64)
            return (120.0, 64, 64)

        prepare_mock = AsyncMock(
            side_effect=AssertionError("manual intake must not split every part")
        )
        generate_mock = AsyncMock(
            side_effect=AssertionError("manual intake must not script before click")
        )
        self.start_patch(
            patch.object(bot, "download_telegram_file", new=fake_telegram_download)
        )
        self.start_patch(
            patch.object(bot, "_extract_source_segment_file", new=fake_extract)
        )
        self.start_patch(
            patch.object(bot, "_prepare_source_segments", new=prepare_mock)
        )
        self.start_patch(
            patch.object(bot, "generate_chunk_script", new=generate_mock)
        )
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))
        self.mock_tg_upload.side_effect = fake_upload

        await bot.handle_telegram_video_job(
            16001,
            88,
            {
                "file_id": "uploaded-source",
                "file_name": "movie.mp4",
                "mime_type": "video/mp4",
                "duration": 250,
                "file_size": 10001,
            },
        )

        self.assertEqual(extract_calls, [(1, 0.0, 120.0)])
        self.assertEqual(events, [("upload", "original_part_0001.mp4")])
        prepare_mock.assert_not_awaited()
        generate_mock.assert_not_awaited()
        self.assertEqual(self.mock_tg_upload.await_count, 1)

        state = bot._find_manual_session_for_chat(16001)
        self.assertIsNotNone(state)
        session_id = str(state["session_id"])
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["current_part_index"], 1)
        self.assertEqual(
            [part["status"] for part in state["parts"]],
            ["ready", "waiting", "waiting"],
        )
        self.assertTrue(state["parts"][0]["source_file"])
        self.assertEqual(
            [part["source_file"] for part in state["parts"][1:]], ["", ""]
        )
        self.assertEqual(
            [part["script_file"] for part in state["parts"]], ["", "", ""]
        )
        self.assertIsNotNone(state["parts"][0]["message_id"])
        self.assertEqual(
            [part["message_id"] for part in state["parts"][1:]], [None, None]
        )
        full_source = Path(state["session_dir"]) / state["full_source_file"]
        self.assertTrue(full_source.exists())
        self.assertFalse((Path(state["session_dir"]) / "full_script.json").exists())

        payload = self.mock_tg_upload.await_args.args[1]
        markup = json.loads(payload["reply_markup"])
        self.assertEqual(markup, bot.manual_part_keyboard(session_id, 1, "ready"))
        self.assertEqual(
            markup["inline_keyboard"][0][0]["callback_data"],
            f"part:recap:{session_id}:1",
        )
        self.assertNotIn(16001, bot.ACTIVE_JOBS)

    async def test_manual_sequential_click_recaps_current_then_sends_next_and_join_is_explicit(self):
        chat_id = int(bot.ADMIN_ID)
        session_id = "manual-test-1"
        state = bot._create_manual_session(
            session_id=session_id,
            chat_id=chat_id,
            reply_msg_id=10,
            status_message_id=11,
            source_spec={"kind": "url", "url": "https://videos.example/manual"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        session_dir = Path(state["session_dir"])
        (session_dir / "source.mp4").write_bytes(b"x" * 10001)
        first_relative = "source_segments/source_part_0001.mp4"
        first_source = session_dir / first_relative
        first_source.parent.mkdir(parents=True, exist_ok=True)
        first_source.write_bytes(b"s" * 100)
        parts = [
            {
                "index": 1,
                "start": 0.0,
                "end": 10.0,
                "duration": 10.0,
                "source_file": first_relative,
                "script_file": "",
                "status": "ready",
                "message_id": 501,
                "recap_file": "",
                "recap_duration_seconds": 0.0,
                "attempts": 0,
                "error": "",
            },
            {
                "index": 2,
                "start": 10.0,
                "end": 20.0,
                "duration": 10.0,
                "source_file": "",
                "script_file": "",
                "status": "waiting",
                "message_id": None,
                "recap_file": "",
                "recap_duration_seconds": 0.0,
                "attempts": 0,
                "error": "",
            },
        ]
        state["source_duration_seconds"] = 20.0
        state["full_source_file"] = "source.mp4"
        state["current_part_index"] = 1
        state["parts"] = parts
        state["status"] = "ready"
        bot._save_manual_session(state)

        script_calls = []
        script_contexts = []
        render_calls = []
        extract_calls = []
        uploaded = []
        concat_calls = []

        async def fake_generate_script(
            _chunk_path,
            actual_dur,
            start,
            end,
            story_context,
            part_num,
            _tracker,
            _settings,
        ):
            script_calls.append(part_num)
            script_contexts.append(dict(story_context))
            return {
                "version": 1,
                "part_index": part_num,
                "global_start": start,
                "global_end": end,
                "duration": actual_dur,
                "scenes": [
                    {
                        "start": 0,
                        "end": actual_dur,
                        "burmese": first_person_text(30),
                    }
                ],
                "cuts": [],
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
                "story_context_before": dict(story_context),
                "story_context_after": {
                    "characters": ["Hero", "Mina"],
                    "summary": f"after {part_num}",
                    "last_scene": f"scene {part_num}",
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                },
                "metrics": {"target_chars": 30, "actual_chars": 30},
            }

        async def fake_render(
            _chunk_path,
            _dur,
            _start,
            _end,
            script,
            part_num,
            _tracker,
            _settings,
            *,
            output_path="",
        ):
            render_calls.append(part_num)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"r" * 100)
            return output_path, script

        def fake_extract(_source_path, output_path, start, duration):
            index = int(Path(output_path).stem.rsplit("_", 1)[-1])
            extract_calls.append((index, float(start), float(duration)))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"s" * 100)
            return True

        async def fake_upload(_method, _payload, _field, file_name, _path, _mime):
            uploaded.append(file_name)
            return {"ok": True, "result": {"message_id": 900 + len(uploaded)}}

        def fake_concat(parts_to_join, output_path, concat_path):
            concat_calls.append(list(parts_to_join))
            Path(concat_path).write_text("\n".join(parts_to_join), encoding="utf-8")
            Path(output_path).write_bytes(b"f" * 100)
            return True

        def fake_probe(path):
            name = Path(path).name
            if name == "final_recap.mp4":
                return (18.0, 64, 64)
            if name.startswith("recap_part_"):
                return (9.0, 64, 64)
            return (10.0, 64, 64)

        self.start_patch(
            patch.object(bot, "generate_chunk_script", new=fake_generate_script)
        )
        self.start_patch(
            patch.object(bot, "render_chunk_recap_from_script", new=fake_render)
        )
        self.start_patch(
            patch.object(bot, "_extract_source_segment_file", new=fake_extract)
        )
        self.start_patch(patch.object(bot, "_concat_video_parts", new=fake_concat))
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))
        self.mock_tg_upload.side_effect = fake_upload

        def callback_update(data, message_id=600):
            return {
                "callback_query": {
                    "id": f"cb-{data}",
                    "data": data,
                    "from": {"id": chat_id},
                    "message": {
                        "message_id": message_id,
                        "chat": {"id": chat_id},
                    },
                }
            }

        # A future part cannot be recapped before the current one.
        self.mock_send_text.reset_mock()
        await bot._handle_manual_part_callback(
            f"part:recap:{session_id}:2", chat_id, 502
        )
        self.assertEqual(script_calls, [])
        self.assertEqual(render_calls, [])
        self.assertEqual(extract_calls, [])
        self.assertEqual(uploaded, [])
        self.assertIn("လက်ရှိပို့ထားတဲ့အပိုင်း", self.mock_send_text.await_args.args[1])

        # Clicking part 1 scripts/renders only part 1, then extracts and
        # sends only original part 2.
        await bot.process_admin_update(
            callback_update(f"part:recap:{session_id}:1", 501)
        )
        task = bot.MANUAL_RENDER_TASKS[session_id]
        await asyncio.wait_for(task, timeout=2)

        self.assertEqual(script_calls, [1])
        self.assertEqual(script_contexts[0]["summary"], "")
        self.assertEqual(render_calls, [1])
        self.assertEqual(extract_calls, [(2, 10.0, 10.0)])
        self.assertEqual(
            uploaded,
            ["recap_part_0001.mp4", "original_part_0002.mp4"],
        )
        state = bot._load_manual_session(session_id)
        self.assertEqual(bot._manual_part(state, 1)["status"], "done")
        self.assertEqual(bot._manual_part(state, 2)["status"], "ready")
        self.assertIsNotNone(bot._manual_part(state, 2)["message_id"])
        self.assertEqual(state["current_part_index"], 2)
        self.assertEqual(state["story_context"]["summary"], "after 1")
        self.assertNotIn(chat_id, bot.ACTIVE_JOBS)

        self.mock_send_text.reset_mock()
        await bot._handle_manual_part_callback(
            f"part:join:{session_id}", chat_id, 601
        )
        self.assertEqual(concat_calls, [])
        self.assertIn(
            "မပြီးသေးပါ", self.mock_send_text.await_args.args[1]
        )

        part_two_message = int(bot._manual_part(state, 2)["message_id"])
        await bot.process_admin_update(
            callback_update(
                f"part:recap:{session_id}:2", part_two_message
            )
        )
        task = bot.MANUAL_RENDER_TASKS[session_id]
        await asyncio.wait_for(task, timeout=2)

        self.assertEqual(script_calls, [1, 2])
        self.assertEqual(script_contexts[1]["summary"], "after 1")
        self.assertEqual(render_calls, [1, 2])
        self.assertEqual(extract_calls, [(2, 10.0, 10.0)])
        self.assertEqual(uploaded[-1], "recap_part_0002.mp4")
        state = bot._load_manual_session(session_id)
        self.assertEqual(state["status"], "ready_to_join")
        self.assertIsNone(state["current_part_index"])
        self.assertEqual(state["story_context"]["summary"], "after 2")

        # The recovery Next button must not extract/send anything after
        # every part is done; it points the user to the explicit join.
        self.mock_send_text.reset_mock()
        await bot._handle_manual_part_callback(
            f"part:next:{session_id}", chat_id, 602
        )
        self.assertEqual(extract_calls, [(2, 10.0, 10.0)])
        self.assertIn("ပြန်ဆက်မယ်", self.mock_send_text.await_args.args[1])

        await bot._handle_manual_part_callback(
            f"part:join:{session_id}", chat_id, 603
        )
        task = bot.MANUAL_RENDER_TASKS[session_id]
        await asyncio.wait_for(task, timeout=2)

        self.assertEqual(
            [Path(path).name for path in concat_calls[0]],
            ["recap_part_0001.mp4", "recap_part_0002.mp4"],
        )
        self.assertEqual(uploaded[-1], "recap_final_1.mp4")
        self.assertIsNone(bot._find_manual_session_for_chat(chat_id))
        self.assertFalse(session_dir.exists())
        self.mock_bump_stat.assert_called_with(chat_id, "done")

    async def test_manual_next_button_retries_failed_next_original(self):
        chat_id = 19001
        session_id = "manual-next-retry"
        state = bot._create_manual_session(
            session_id=session_id,
            chat_id=chat_id,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={"kind": "url", "url": "https://videos.example/next"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        session_dir = Path(state["session_dir"])
        (session_dir / "source.mp4").write_bytes(b"x" * 10001)
        state["full_source_file"] = "source.mp4"
        state["current_part_index"] = 2
        state["status"] = "failed"
        state["parts"] = [
            {
                "index": 1,
                "start": 0.0,
                "end": 10.0,
                "duration": 10.0,
                "source_file": "",
                "script_file": "",
                "status": "done",
                "message_id": 501,
                "recap_file": "recaps/recap_part_0001.mp4",
                "recap_duration_seconds": 9.0,
                "attempts": 1,
                "error": "",
            },
            {
                "index": 2,
                "start": 10.0,
                "end": 20.0,
                "duration": 10.0,
                "source_file": "",
                "script_file": "",
                "status": "extract_failed",
                "message_id": None,
                "recap_file": "",
                "recap_duration_seconds": 0.0,
                "attempts": 0,
                "error": "previous extraction interrupted",
            },
        ]
        bot._save_manual_session(state)

        extract_calls = []
        uploaded = []

        def fake_extract(_source_path, output_path, start, duration):
            extract_calls.append((float(start), float(duration)))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"s" * 100)
            return True

        async def fake_upload(_method, _payload, _field, file_name, _path, _mime):
            uploaded.append(file_name)
            return {"ok": True, "result": {"message_id": 502}}

        self.start_patch(
            patch.object(bot, "_extract_source_segment_file", new=fake_extract)
        )
        self.start_patch(
            patch.object(bot, "probe_video_info", return_value=(10.0, 64, 64))
        )
        self.mock_tg_upload.side_effect = fake_upload

        await bot._handle_manual_part_callback(
            f"part:next:{session_id}", chat_id, 700
        )
        task = bot.MANUAL_RENDER_TASKS[session_id]
        await asyncio.wait_for(task, timeout=2)

        self.assertEqual(extract_calls, [(10.0, 10.0)])
        self.assertEqual(uploaded, ["original_part_0002.mp4"])
        state = bot._load_manual_session(session_id)
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["current_part_index"], 2)
        self.assertEqual(bot._manual_part(state, 2)["status"], "ready")
        self.assertEqual(bot._manual_part(state, 2)["message_id"], 502)
        self.assertNotIn(chat_id, bot.ACTIVE_JOBS)

    async def test_share_text_extracts_rednote_url(self):
        chat_id = int(bot.ADMIN_ID)
        fake_link_job = AsyncMock(return_value=None)
        self.start_patch(patch.object(bot, "handle_link_job", new=fake_link_job))
        await bot.process_admin_update(
            {
                "message": {
                    "message_id": 77,
                    "chat": {"id": chat_id},
                    "from": {"id": chat_id},
                    "text": "ဒီဟာလေး https://xhslink.com/a/example123 ကြည့်ပါ",
                }
            }
        )
        fake_link_job.assert_awaited_once_with(
            chat_id, 77, "https://xhslink.com/a/example123"
        )

    @unittest.skipUnless(
        os.path.exists(bot.FFMPEG_BIN) or shutil.which(bot.FFMPEG_BIN),
        "ffmpeg is not available",
    )
    async def test_render_from_prepared_script_does_not_regenerate_script(self):
        chunk_path = self.tmp_path / "prepared_chunk.mp4"
        result = subprocess.run(
            [
                bot.FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=purple:s=64x64:r=25:d=3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(chunk_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest("could not generate a local MP4 fixture")

        def forbidden_upload(file):
            raise AssertionError("prepared script must not upload for script generation")

        def forbidden_generate(model, contents, config):
            raise AssertionError("prepared script must not be regenerated")

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=forbidden_upload,
                get=lambda name: None,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=forbidden_generate),
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )

        async def fake_tts(_scenes, _work_dir, _settings):
            return [
                np.full(int(bot.SAMPLE_RATE * 2.7), 0.5, dtype=np.float32)
            ]

        self.start_patch(patch.object(bot, "synthesize_tts", new=fake_tts))
        prepared = {
            "version": 1,
            "part_index": 1,
            "scenes": [
                {"start": 0, "end": 3, "burmese": first_person_text(27)}
            ],
            "cuts": [],
            "protagonist": "Hero",
            "pronoun": MALE_PRONOUN,
            "story_context_before": {},
            "story_context_after": {
                "characters": ["Hero"],
                "summary": "done",
                "last_scene": "end",
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
            },
            "metrics": {"target_chars": 27},
        }
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))
        output_path = self.tmp_path / "prepared_recap.mp4"
        rendered, updated = await bot.render_chunk_recap_from_script(
            str(chunk_path),
            3,
            0,
            3,
            prepared,
            1,
            tracker,
            settings={
                "coverage": 90,
                "rate": 10.0,
                "speed": 1.0,
                "pause": 0,
                "voice": "Charon",
                "resolution": 64,
                "shield": False,
                "color": False,
                "quality": 35,
                "fast_mode": True,
            },
            output_path=str(output_path),
        )

        self.assertEqual(rendered, str(output_path))
        self.assertTrue(chunk_path.exists())
        self.assertAlmostEqual(
            bot.probe_video_info(rendered)[0], 2.7, delta=0.2
        )
        self.assertIn("rendered_duration_seconds", updated["metrics"])

    def test_stale_manual_render_is_reset_after_restart(self):
        state = bot._create_manual_session(
            session_id="manual-stale-1",
            chat_id=18001,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={"kind": "url", "url": "https://videos.example/stale"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        state["parts"] = [
            {"index": 1, "status": "rendering", "start": 0, "end": 10}
        ]
        state["status"] = "rendering"
        state["active_render_part"] = 1
        state["cancel_requested"] = True
        bot.CANCELLED_JOBS.add("manual-stale-1")
        bot._save_manual_session(state)

        self.assertEqual(bot._reset_stale_manual_sessions_after_restart(), 1)
        reset = bot._load_manual_session("manual-stale-1")
        self.assertEqual(reset["status"], "ready")
        self.assertEqual(reset["parts"][0]["status"], "failed")
        self.assertFalse(reset["cancel_requested"])
        self.assertNotIn("manual-stale-1", bot.CANCELLED_JOBS)

    def test_stale_manual_next_part_is_reset_after_restart(self):
        state = bot._create_manual_session(
            session_id="manual-stale-next",
            chat_id=18002,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={"kind": "url", "url": "https://videos.example/stale-next"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        state["parts"] = [
            {"index": 1, "status": "done", "start": 0, "end": 10},
            {"index": 2, "status": "extracting", "start": 10, "end": 20},
        ]
        state["status"] = "sending_next"
        state["current_part_index"] = 2
        bot._save_manual_session(state)

        self.assertEqual(bot._reset_stale_manual_sessions_after_restart(), 1)
        reset = bot._load_manual_session("manual-stale-next")
        self.assertEqual(reset["status"], "ready")
        self.assertEqual(reset["parts"][1]["status"], "extract_failed")
        self.assertEqual(reset["current_part_index"], 2)

    def test_manual_recovery_token_roundtrip_and_budget(self):
        state = bot._create_manual_session(
            session_id="manual-recover-token",
            chat_id=18101,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={
                "kind": "telegram_file",
                "file_id": "SOURCEFILEID1234567890",
                "file_name": "movie.mp4",
                "file_size": 15000000,
                "mime_type": "video/mp4",
            },
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        # The Telegram file reference must persist so the source can be
        # re-acquired after the ephemeral session disk is wiped.
        self.assertEqual(state["source"]["file_id"], "SOURCEFILEID1234567890")
        self.assertEqual(state["source"]["file_size"], 15000000)

        state["source_duration_seconds"] = 250.0
        state["parts"] = [
            {
                "index": 1,
                "start": 0.0,
                "end": 120.0,
                "duration": 120.0,
                "status": "done",
                "recap_file_id": "RECAPFILEID0001",
            },
            {
                "index": 2,
                "start": 120.0,
                "end": 240.0,
                "duration": 120.0,
                "status": "ready",
            },
            {
                "index": 3,
                "start": 240.0,
                "end": 250.0,
                "duration": 10.0,
                "status": "waiting",
            },
        ]
        state["story_context"] = {
            "characters": ["Hero", "Mina"],
            "summary": "ဇာတ်လမ်းအကျဉ်း " * 120,
            "last_scene": "နောက်ဆုံးခန်း " * 80,
            "protagonist": "Hero",
            "pronoun": MALE_PRONOUN,
        }
        state["status"] = "ready"
        bot._save_manual_session(state)

        token = bot._build_manual_recovery_token(
            state, anchor_index=2, max_token_chars=620
        )
        self.assertTrue(token)
        self.assertLessEqual(len(token), 620)

        payload = bot._manual_recovery_decode(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sid"], "manual-recover-token")
        self.assertEqual(payload["k"], "telegram_file")
        self.assertEqual(payload["f"], "SOURCEFILEID1234567890")
        self.assertEqual(payload["d"], 250.0)
        self.assertEqual(payload["i"], 2)
        self.assertEqual(payload["done"], [1])
        self.assertEqual(payload["r"], {"1": "RECAPFILEID0001"})
        self.assertEqual(payload["ctx"]["p"], "Hero")
        self.assertEqual(payload["ctx"]["pr"], MALE_PRONOUN)

        caption = f"🎬 မူရင်းအပိုင်း 2/3 (02:00–04:00)\n♻️ RNB1.{token}"
        from_caption = bot._extract_manual_recovery_payload(caption)
        self.assertIsNotNone(from_caption)
        self.assertEqual(from_caption["sid"], "manual-recover-token")
        self.assertIn("RNB1.", bot._manual_status_text(state))

    async def test_rebuild_manual_session_from_payload_restores_everything(self):
        chat_id = 18102
        session_id = "manual-recover-build"
        payload = {
            "v": 1,
            "sid": session_id,
            "k": "url",
            "u": "https://videos.example/recover",
            "d": 250.0,
            "i": 2,
            "done": [1],
            "r": {"1": "RECAPFILEID0001"},
            "ctx": {
                "c": ["Hero", "Mina"],
                "s": "before part two",
                "l": "door opens",
                "p": "Hero",
                "pr": MALE_PRONOUN,
            },
        }
        extract_calls = []

        async def fake_acquire(_spec, work_dir, _tracker=None):
            path = Path(work_dir) / "source.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 10001)
            return str(path), 250.0

        async def fake_download(_file_id, out_path, _tracker=None, **_kwargs):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"r" * 100)
            return 108.0

        def fake_extract(_source_path, output_path, start, duration):
            extract_calls.append((float(start), float(duration)))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"s" * 100)
            return True

        def fake_probe(path):
            name = Path(path).name
            if name.startswith("recap_part_"):
                return (108.0, 64, 64)
            return (120.0, 64, 64)

        self.start_patch(patch.object(bot, "_acquire_source_file", new=fake_acquire))
        self.start_patch(
            patch.object(bot, "download_telegram_file", new=fake_download)
        )
        self.start_patch(
            patch.object(bot, "_extract_source_segment_file", new=fake_extract)
        )
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))

        state = await bot._rebuild_manual_session_from_payload(
            payload,
            chat_id,
            tracker=bot._NullProgressTracker(),
            clicked_part_message_id=777,
            need_current_segment=True,
        )

        parts = bot._manual_sorted_parts(state)
        self.assertEqual([p["status"] for p in parts], ["done", "ready", "waiting"])
        session_dir = Path(state["session_dir"])
        self.assertTrue((session_dir / "source.mp4").exists())
        self.assertTrue(
            (session_dir / "recaps" / "recap_part_0001.mp4").exists()
        )
        self.assertEqual(parts[0]["recap_file_id"], "RECAPFILEID0001")
        self.assertEqual(parts[0]["recap_duration_seconds"], 108.0)
        self.assertTrue(
            (session_dir / "source_segments" / "source_part_0002.mp4").exists()
        )
        self.assertEqual(parts[1]["message_id"], 777)
        self.assertEqual(extract_calls, [(120.0, 120.0)])
        self.assertEqual(state["story_context"]["protagonist"], "Hero")
        self.assertEqual(state["story_context"]["summary"], "before part two")
        self.assertEqual(state["current_part_index"], 2)
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["recovery_count"], 1)
        reloaded = bot._load_manual_session(session_id)
        self.assertEqual(reloaded["source"]["url"], "https://videos.example/recover")

    async def test_missing_session_click_recovers_from_caption_then_recaps(self):
        chat_id = 18103
        session_id = "manual-recover-click"
        token = bot._manual_recovery_encode(
            {
                "v": 1,
                "sid": session_id,
                "k": "url",
                "u": "https://videos.example/click-recover",
                "d": 120.0,
                "i": 1,
            }
        )
        rebuild_calls = []

        async def fake_rebuild(payload, rebuild_chat_id, tracker=None, **_kwargs):
            rebuild_calls.append(payload["sid"])
            state = bot._create_manual_session(
                session_id=session_id,
                chat_id=rebuild_chat_id,
                reply_msg_id=1,
                status_message_id=2,
                source_spec={"kind": "url", "url": payload["u"]},
                settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
            )
            session_dir = Path(state["session_dir"])
            relative = "source_segments/source_part_0001.mp4"
            segment = session_dir / relative
            segment.parent.mkdir(parents=True, exist_ok=True)
            segment.write_bytes(b"s" * 100)
            state["source_duration_seconds"] = 120.0
            state["full_source_file"] = "source.mp4"
            state["parts"] = [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 120.0,
                    "duration": 120.0,
                    "source_file": relative,
                    "script_file": "",
                    "status": "ready",
                    "message_id": 501,
                    "recap_file": "",
                    "recap_duration_seconds": 0.0,
                    "attempts": 0,
                    "error": "",
                }
            ]
            state["current_part_index"] = 1
            state["status"] = "ready"
            bot._save_manual_session(state)
            return state

        fake_render = AsyncMock(return_value=None)
        self.start_patch(
            patch.object(
                bot, "_rebuild_manual_session_from_payload", new=fake_rebuild
            )
        )
        self.start_patch(
            patch.object(bot, "_run_manual_part_recap", new=fake_render)
        )
        self.start_patch(
            patch.object(
                bot, "edit_reply_markup", new=AsyncMock(return_value=None)
            )
        )
        message = {
            "message_id": 501,
            "chat": {"id": chat_id},
            "caption": f"🎬 မူရင်းအပိုင်း 1/1 (00:00–02:00)\n♻️ RNB1.{token}",
            "video": {"file_id": "PARTFILEID"},
        }
        try:
            await bot._handle_manual_part_callback(
                f"part:recap:{session_id}:1", chat_id, 501, message=message
            )
            for _ in range(20):
                pending = [
                    task for task in list(bot.BACKGROUND_TASKS) if not task.done()
                ]
                if not pending:
                    break
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            bot.MANUAL_RECOVERY_IN_PROGRESS.discard(session_id)

        self.assertEqual(rebuild_calls, [session_id])
        fake_render.assert_awaited_once()
        self.assertEqual(fake_render.await_args.args[0], session_id)
        self.assertEqual(fake_render.await_args.args[1], 1)
        self.assertNotIn(session_id, bot.MANUAL_RECOVERY_IN_PROGRESS)

    async def test_legacy_caption_rebuild_recovers_current_part(self):
        chat_id = 18104
        message = {
            "message_id": 501,
            "chat": {"id": chat_id},
            "caption": "🎬 မူရင်းအပိုင်း 1/2 (00:00\u201302:00)\n🎬 Recap လုပ်မယ် ကို နှိပ်ပါ။",
            "video": {"file_id": "OLDPARTFILE", "file_size": 1000},
        }
        info = bot._legacy_manual_caption_info(message)
        self.assertIsNotNone(info)
        self.assertEqual(info["index"], 1)
        self.assertEqual(info["total"], 2)
        self.assertEqual(info["start"], 0.0)
        self.assertEqual(info["end"], 120.0)

        async def fake_download(_file_id, out_path, _tracker=None, **_kwargs):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"s" * 100)
            return 120.0

        self.start_patch(
            patch.object(bot, "download_telegram_file", new=fake_download)
        )
        self.start_patch(
            patch.object(bot, "probe_video_info", new=lambda _p: (120.0, 64, 64))
        )

        state = await bot._rebuild_manual_session_from_legacy_message(
            "manual-legacy-1",
            chat_id,
            message,
            1,
            tracker=bot._NullProgressTracker(),
        )
        parts = bot._manual_sorted_parts(state)
        self.assertEqual([p["status"] for p in parts], ["ready", "waiting"])
        self.assertEqual(parts[0]["message_id"], 501)
        segment = (
            Path(state["session_dir"])
            / "source_segments"
            / "source_part_0001.mp4"
        )
        self.assertTrue(segment.exists())
        self.assertTrue(state["legacy_partial"])
        self.assertEqual(state["current_part_index"], 1)
        self.assertEqual(state["source_duration_seconds"], 240.0)

    async def test_ensure_manual_full_source_redownloads_when_missing(self):
        state = bot._create_manual_session(
            session_id="manual-ensure-source",
            chat_id=18105,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={"kind": "url", "url": "https://videos.example/ensure"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        state["full_source_file"] = "source.mp4"
        bot._save_manual_session(state)
        acquire_calls = []

        async def fake_acquire(_spec, work_dir, _tracker=None):
            acquire_calls.append(work_dir)
            path = Path(work_dir) / "source.mp4"
            path.write_bytes(b"x" * 100)
            return str(path), 250.0

        self.start_patch(patch.object(bot, "_acquire_source_file", new=fake_acquire))
        restored = await bot._ensure_manual_full_source(state)
        self.assertTrue(restored and Path(restored).exists())
        self.assertEqual(len(acquire_calls), 1)
        self.assertEqual(state["source_duration_seconds"], 250.0)

        # A second call reuses the restored file without downloading again.
        again = await bot._ensure_manual_full_source(state)
        self.assertEqual(again, restored)
        self.assertEqual(len(acquire_calls), 1)

    async def test_expired_session_is_revived_on_button_tap_when_files_survive(self):
        chat_id = 18106
        session_id = "manual-expired-revive"
        state = bot._create_manual_session(
            session_id=session_id,
            chat_id=chat_id,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={"kind": "url", "url": "https://videos.example/revive"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        state["source_duration_seconds"] = 120.0
        state["parts"] = [
            {
                "index": 1,
                "start": 0.0,
                "end": 120.0,
                "duration": 120.0,
                "status": "ready",
                "message_id": 501,
            }
        ]
        state["status"] = "ready"
        state["current_part_index"] = 1
        state["expires_at_unix"] = 1.0
        bot._save_manual_session(state)
        self.assertTrue(bot._manual_session_expired(state))

        self.mock_send_text.reset_mock()
        await bot._handle_manual_part_callback(
            f"part:status:{session_id}", chat_id, 601, message={}
        )

        sent_text = self.mock_send_text.await_args.args[1]
        self.assertIn("၂ မိနစ်အပိုင်းများ အခြေအနေ", sent_text)
        self.assertNotIn("သက်တမ်းကုန်", sent_text)
        revived = bot._load_manual_session(session_id)
        self.assertGreater(float(revived["expires_at_unix"]), time.time())

    async def test_guarded_update_reports_retryable_error_to_user(self):
        failing = AsyncMock(side_effect=RuntimeError("boom"))
        self.start_patch(patch.object(bot, "process_admin_update", new=failing))
        answer = AsyncMock(return_value=None)
        self.start_patch(patch.object(bot, "answer_callback", new=answer))
        self.mock_send_text.reset_mock()
        update = {
            "callback_query": {
                "id": "cb-err",
                "data": "part:recap:whatever:1",
                "from": {"id": 18201},
                "message": {"message_id": 10, "chat": {"id": 18201}},
            }
        }

        await bot._process_admin_update_guarded(update)

        failing.assert_awaited_once()
        answer.assert_awaited_once()
        self.assertEqual(answer.await_args.args[0], "cb-err")
        sent_chat, sent_text = self.mock_send_text.await_args.args[:2]
        self.assertEqual(sent_chat, 18201)
        self.assertIn("ယာယီအမှား", sent_text)
        self.assertIn("ထပ်နှိပ်ကြည့်ပါ", sent_text)

    async def test_original_part_is_sent_as_downloadable_copy_when_large(self):
        chat_id = 18202
        state = bot._create_manual_session(
            session_id="manual-send-cap",
            chat_id=chat_id,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={"kind": "url", "url": "https://videos.example/big"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        session_dir = Path(state["session_dir"])
        relative = "source_segments/source_part_0001.mp4"
        segment = session_dir / relative
        segment.parent.mkdir(parents=True, exist_ok=True)
        with open(segment, "wb") as handle:
            handle.truncate(bot.MANUAL_RECOVERABLE_SEND_BYTES + 1024)
        state["source_duration_seconds"] = 120.0
        part = {
            "index": 1,
            "start": 0.0,
            "end": 120.0,
            "duration": 120.0,
            "source_file": relative,
            "status": "ready",
            "message_id": None,
        }
        state["parts"] = [part]
        bot._save_manual_session(state)

        preview_calls = []

        def fake_preview(_source_path, output_path, target_bytes=None):
            preview_calls.append(target_bytes)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"p" * 100)
            return True

        uploaded_paths = []

        async def fake_upload(_method, payload, _field, _name, file_path, _mime):
            uploaded_paths.append((file_path, payload.get("caption") or ""))
            return {
                "ok": True,
                "result": {
                    "message_id": 555,
                    "video": {"file_id": "SENTPARTFILE"},
                },
            }

        self.start_patch(
            patch.object(bot, "_make_telegram_preview_copy", new=fake_preview)
        )
        self.mock_tg_upload.side_effect = fake_upload

        message_id = await bot._send_manual_original_part(state, part)

        self.assertEqual(message_id, 555)
        self.assertEqual(len(preview_calls), 1)
        self.assertLessEqual(
            preview_calls[0], bot.MANUAL_RECOVERABLE_SEND_BYTES
        )
        self.assertIn("previews", uploaded_paths[0][0])
        self.assertIn("RNB1.", uploaded_paths[0][1])
        self.assertLessEqual(
            bot._telegram_text_units(uploaded_paths[0][1]), 1024
        )
        self.assertEqual(part["source_telegram_file_id"], "SENTPARTFILE")
        self.assertTrue(part["sent_preview"])

    def test_manual_status_text_stays_within_telegram_limit_for_many_parts(self):
        state = bot._create_manual_session(
            session_id="manual-many-parts",
            chat_id=18203,
            reply_msg_id=1,
            status_message_id=2,
            source_spec={"kind": "url", "url": "https://videos.example/long"},
            settings=bot._with_required_coverage(bot.DEFAULT_SETTINGS),
        )
        state["source_duration_seconds"] = 900 * 120.0
        state["parts"] = [
            {
                "index": number,
                "start": (number - 1) * 120.0,
                "end": number * 120.0,
                "duration": 120.0,
                "status": "waiting",
            }
            for number in range(1, 901)
        ]
        state["status"] = "ready"
        bot._save_manual_session(state)

        text = bot._manual_status_text(state)
        self.assertLessEqual(bot._telegram_text_units(text), 4096)
        self.assertIn("(+860)", text)
        self.assertIn("RNB1.", text)

    async def test_n8n_forwarding_failure_falls_back_to_direct_pipeline(self):
        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", False))
        self.start_patch(patch.object(bot, "PIPELINE_MODE", "n8n"))
        self.start_patch(
            patch.object(bot, "N8N_TELEGRAM_HANDOFF_ENABLED", True)
        )
        self.start_patch(patch.object(bot, "AUTO_SOURCE_ATTEMPTS", 1))
        self.start_patch(
            patch.object(bot, "N8N_WEBHOOK_URL", "https://n8n.example/webhook")
        )
        self.start_patch(
            patch.object(bot.secrets, "token_hex", Mock(return_value="abc123def0"))
        )
        forward = AsyncMock(return_value=False)
        self.start_patch(patch.object(bot, "_forward_job_to_n8n", new=forward))
        direct_entry = AsyncMock(side_effect=RuntimeError("stop direct test here"))
        self.start_patch(patch.object(bot, "_acquire_source_file", new=direct_entry))

        await bot.handle_link_job(14001, 55, "https://videos.example/fallback")

        forward.assert_awaited_once()
        forwarded_state = forward.await_args.args[0]
        self.assertEqual(forwarded_state["job_id"], "jabc123def0")
        self.assertFalse((self.jobs_dir / "jabc123def0").exists())
        direct_entry.assert_awaited_once()
        self.assertEqual(
            direct_entry.await_args.args[0],
            {"kind": "url", "url": "https://videos.example/fallback"},
        )
        self.assertNotIn(14001, bot.ACTIVE_JOBS)

    async def test_manual_flow_requires_double_opt_in(self):
        resolve = bot._resolve_manual_parts_enabled
        self.assertFalse(resolve("manual", ""))
        self.assertFalse(resolve("manual", "0"))
        self.assertTrue(resolve("manual", "1"))
        self.assertTrue(resolve("sequential", "yes"))
        self.assertTrue(resolve("MANUAL", "ON"))
        self.assertFalse(resolve("auto", "1"))
        self.assertFalse(resolve("", "1"))
        self.assertFalse(resolve("auto", ""))

        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", False))
        self.start_patch(patch.object(bot, "PIPELINE_MODE", "n8n"))
        self.start_patch(
            patch.object(bot, "N8N_TELEGRAM_HANDOFF_ENABLED", False)
        )
        # A stale PIPELINE_MODE=n8n must not reroute Telegram jobs.
        self.assertEqual(bot._telegram_pipeline_name(), "direct_auto")

    async def test_telegram_pipeline_name_variants(self):
        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", False))
        self.start_patch(patch.object(bot, "PIPELINE_MODE", "n8n"))
        self.start_patch(
            patch.object(bot, "N8N_TELEGRAM_HANDOFF_ENABLED", True)
        )
        self.assertEqual(bot._telegram_pipeline_name(), "n8n_handoff_opt_in")

        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", True))
        self.assertEqual(bot._telegram_pipeline_name(), "direct_manual_opt_in")

    async def test_auto_segment_recap_retries_transient_failure(self):
        self.start_patch(
            patch.object(bot.asyncio, "sleep", new=AsyncMock(return_value=None))
        )
        build = AsyncMock(
            side_effect=[
                RuntimeError("ffmpeg died transiently"),
                ("recap.mp4", {"summary": "ok"}),
            ]
        )
        self.start_patch(patch.object(bot, "build_chunk_recap", new=build))
        cleanup = Mock()
        self.start_patch(
            patch.object(bot, "_cleanup_auto_segment_retry_artifacts", new=cleanup)
        )

        recap, summary = await bot._build_chunk_recap_with_retry(
            "chunk.mp4", 120.0, 0.0, 120.0, {}, 1, None, {}, total_parts=3
        )

        self.assertEqual(recap, "recap.mp4")
        self.assertEqual(summary, {"summary": "ok"})
        self.assertEqual(build.await_count, 2)
        cleanup.assert_called_once_with("chunk.mp4")

    async def test_auto_source_acquisition_retries_only_transient_failures(self):
        self.start_patch(
            patch.object(bot.asyncio, "sleep", new=AsyncMock(return_value=None))
        )
        acquire = AsyncMock(
            side_effect=[RuntimeError("read timeout"), ("src.mp4", 30.0)]
        )
        self.start_patch(patch.object(bot, "_acquire_source_file", new=acquire))

        result = await bot._acquire_source_file_with_retry(
            {"kind": "url", "url": "https://videos.example/x"},
            str(self.tmp_path),
            None,
        )
        self.assertEqual(result, ("src.mp4", 30.0))
        self.assertEqual(acquire.await_count, 2)

        permanent = AsyncMock(
            side_effect=bot.SourceDownloadError(
                "too big", detail="Telegram Bot API 20MB download limit"
            )
        )
        self.start_patch(patch.object(bot, "_acquire_source_file", new=permanent))
        with self.assertRaises(bot.SourceDownloadError):
            await bot._acquire_source_file_with_retry(
                {"kind": "url", "url": "https://videos.example/big"},
                str(self.tmp_path),
                None,
            )
        self.assertEqual(permanent.await_count, 1)

    async def test_auto_partial_final_upload_is_failure_not_success(self):
        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", False))
        self.start_patch(
            patch.object(bot.asyncio, "sleep", new=AsyncMock(return_value=None))
        )

        async def fake_telegram_download(_file_id, output_path, _tracker=None, **_kwargs):
            Path(output_path).write_bytes(b"x" * 10001)
            return 60.0

        async def fake_prepare(_source_path, work_dir, _duration, _seconds, tracker=None, **_kwargs):
            path = Path(work_dir) / "source_part_0001.mp4"
            path.write_bytes(b"s" * 100)
            return [
                {
                    "index": 1,
                    "path": str(path),
                    "start": 0.0,
                    "end": 60.0,
                    "duration": 60.0,
                }
            ]

        async def fake_build(chunk_path, *_args, **_kwargs):
            summary = {
                "characters": ["Hero"],
                "summary": "done",
                "last_scene": "end",
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
            }
            if _kwargs.get("script_only"):
                return {
                    "scenes": [
                        {
                            "start": 0.0,
                            "end": 10.0,
                            "burmese": f"{MALE_PRONOUN} ပြောပြတယ်။",
                        }
                    ],
                    "cuts": [],
                    "protagonist": "Hero",
                    "pronoun": MALE_PRONOUN,
                    "story_context_after": summary,
                    "metrics": {"target_chars": 100},
                }, summary
            output = Path(chunk_path).parent / "recap_part_0001.mp4"
            output.write_bytes(b"r" * 100)
            return str(output), summary

        def fake_concat(parts, output_path, concat_path):
            Path(concat_path).write_text("\n".join(parts), encoding="utf-8")
            Path(output_path).write_bytes(b"joined" * 100)
            return True

        def fake_probe(path):
            if Path(path).name == "final_recap.mp4":
                return (54.0, 64, 64)
            return (60.0, 64, 64)

        async def fake_split(path, work_dir):
            extra = Path(work_dir) / "final_part2.mp4"
            extra.write_bytes(b"p2" * 100)
            return [path, str(extra)]

        self.start_patch(
            patch.object(bot, "download_telegram_file", new=fake_telegram_download)
        )
        self.start_patch(
            patch.object(bot, "_prepare_source_segments", new=fake_prepare)
        )
        self.start_patch(patch.object(bot, "build_chunk_recap", new=fake_build))
        self.start_patch(patch.object(bot, "_concat_video_parts", new=fake_concat))
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))
        self.start_patch(
            patch.object(bot, "split_for_telegram", new=AsyncMock(side_effect=fake_split))
        )
        # First final part uploads; the second fails on every attempt.
        self.mock_tg_upload.reset_mock()
        self.mock_tg_upload.side_effect = [{"ok": True}] + [{"ok": False}] * 3
        self.mock_bump_stat.reset_mock()

        await bot.handle_telegram_video_job(
            16001,
            99,
            {
                "file_id": "uploaded-source",
                "file_name": "movie.mp4",
                "mime_type": "video/mp4",
                "duration": 60,
                "file_size": 10001,
            },
        )

        # 1 successful upload + 3 attempts on the failing part.
        self.assertEqual(self.mock_tg_upload.await_count, 4)
        bumped = [call.args for call in self.mock_bump_stat.call_args_list]
        self.assertIn((16001, "failed"), bumped)
        self.assertNotIn((16001, "done"), bumped)
        self.assertNotIn(16001, bot.ACTIVE_JOBS)

    async def test_prepare_source_segments_accepts_keyframe_snapped_boundaries(self):
        # Real files almost never have keyframes exactly on 120s marks.
        # Snapped boundaries (each part <= 1.5x the window, full coverage)
        # must be accepted from the one-pass stream copy instead of
        # forcing a full re-encode extraction (the "frozen at precisely
        # splitting part 3/4" bug).
        work_dir = self.tmp_path / "segment-snapped"
        work_dir.mkdir()

        def fake_split(_path, out_dir, _segment_seconds):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            results = []
            for index in range(3):
                part = out / f"part_{index:04d}.mp4"
                part.write_bytes(b"x" * 100)
                results.append(str(part))
            return results

        durations = {
            "part_0000.mp4": 2.6,
            "part_0001.mp4": 1.5,
            "part_0002.mp4": 0.9,
        }

        def fake_probe(path):
            return (durations[Path(path).name], 64, 64)

        extract = Mock(side_effect=AssertionError("extract must not run"))
        self.start_patch(patch.object(bot, "_split_video_once", new=fake_split))
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))
        self.start_patch(
            patch.object(bot, "_extract_source_segment_file", new=extract)
        )

        segments = await bot._prepare_source_segments(
            str(self.tmp_path / "source.mp4"),
            str(work_dir),
            5,
            segment_seconds=2,
            progress_interval=0.01,
        )

        self.assertEqual([item["duration"] for item in segments], [2.6, 1.5, 0.9])
        self.assertEqual([item["start"] for item in segments], [0.0, 2.6, 4.1])
        self.assertEqual([item["end"] for item in segments], [2.6, 4.1, 5.0])
        extract.assert_not_called()
        manifest = json.loads((work_dir / "source_segments.json").read_text())
        self.assertEqual(manifest["split_mode"], "stream_copy")

    async def test_prepare_source_segments_falls_back_when_part_exceeds_1p5x(self):
        work_dir = self.tmp_path / "segment-oversized"
        work_dir.mkdir()
        extract_calls = []

        def fake_split(_path, out_dir, _segment_seconds):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            results = []
            for index in range(2):
                part = out / f"part_{index:04d}.mp4"
                part.write_bytes(b"x" * 100)
                results.append(str(part))
            return results

        def fake_extract(_source, output_path, start, duration, **kwargs):
            extract_calls.append((start, duration, kwargs))
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 100)
            return True

        def fake_probe(path):
            name = Path(path).name
            if name.startswith("source_part"):
                return (1.0 if name.endswith("0003.mp4") else 2.0, 64, 64)
            return (3.5 if name == "part_0000.mp4" else 1.5, 64, 64)

        self.start_patch(patch.object(bot, "_split_video_once", new=fake_split))
        self.start_patch(patch.object(bot, "probe_video_info", new=fake_probe))
        self.start_patch(
            patch.object(bot, "_extract_source_segment_file", new=fake_extract)
        )

        segments = await bot._prepare_source_segments(
            str(self.tmp_path / "source.mp4"),
            str(work_dir),
            5,
            segment_seconds=2,
            progress_interval=0.01,
        )

        self.assertEqual([item["index"] for item in segments], [1, 2, 3])
        self.assertEqual(len(extract_calls), 3)
        # The fallback must use the bounded cheap profile, not full-res.
        for _start, _duration, kwargs in extract_calls:
            self.assertEqual(kwargs.get("max_height"), 720)
            self.assertEqual(kwargs.get("preset"), "ultrafast")
        manifest = json.loads((work_dir / "source_segments.json").read_text())
        self.assertEqual(manifest["split_mode"], "accurate_reencode")

    async def test_keep_awake_pings_public_healthz_only_while_busy(self):
        calls = []

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, params=None):
                calls.append(url)
                return SimpleNamespace(status_code=200)

        self.start_patch(
            patch.object(bot, "PUBLIC_BASE_URL", "https://example.test/")
        )
        self.start_patch(
            patch.object(bot.httpx, "AsyncClient", new=FakeAsyncClient)
        )

        # Idle: no ping.
        task = asyncio.create_task(bot._keep_awake_while_working())
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(calls, [])

        # Busy: pings the public /healthz URL once per interval.
        self.active_jobs[111] = "job-x"
        task = asyncio.create_task(bot._keep_awake_while_working())
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(calls, ["https://example.test/healthz"])

    async def test_direct_gemini_json_times_out_per_model_and_heartbeats(self):
        # Regression: a single hanging generate_content call used to
        # freeze the job forever at "retrying script (3/3)".
        updates = []

        async def fake_update(text, force=False):
            updates.append(text)

        tracker = SimpleNamespace(update=fake_update)
        called_models = []

        def hang(model, contents, config):
            called_models.append(model)
            time.sleep(1.0)
            return SimpleNamespace(text="{}")

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=hang)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self.start_patch(
            patch.object(bot, "GEMINI_GENERATE_TIMEOUT_SECONDS", 0.4)
        )
        self.start_patch(patch.object(bot, "GEMINI_HEARTBEAT_SECONDS", 0.05))
        self.start_patch(
            patch.object(bot.asyncio, "sleep", new=AsyncMock(return_value=None))
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )

        started = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            await bot._generate_direct_gemini_json(
                fake_client, ["contents"], purpose="test script", tracker=tracker
            )
        elapsed = time.monotonic() - started

        self.assertIn("failed after 2 attempts", str(ctx.exception))
        self.assertEqual(called_models, ["m1", "m2"])
        # Bounded: ~0.4s per attempt, not the full 1s hang per model.
        self.assertLess(elapsed, 1.8)
        self.assertTrue(any("ရေးနေဆဲ" in text for text in updates))

    async def test_direct_gemini_json_succeeds_within_timeout(self):
        payload = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "ကျွန်တော်"}]}
        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda model, contents, config: SimpleNamespace(
                    text=json.dumps(payload, ensure_ascii=False)
                )
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )

        data = await bot._generate_direct_gemini_json(
            fake_client, ["contents"], purpose="test script", tracker=None
        )
        self.assertEqual(data, payload)

    async def test_script_error_is_not_retried_by_segment_wrapper(self):
        build = AsyncMock(side_effect=bot.GeminiScriptError("no scenes"))
        self.start_patch(patch.object(bot, "build_chunk_recap", new=build))

        with self.assertRaises(bot.GeminiScriptError):
            await bot._build_chunk_recap_with_retry(
                "chunk.mp4", 120.0, 0.0, 120.0, {}, 1, None, {}, total_parts=2
            )
        # Inner generation already exhausted every model fallback; the
        # whole-segment wrapper must fail fast instead of re-uploading.
        self.assertEqual(build.await_count, 1)

    async def test_default_voice_is_original_charon_without_deep_filter(self):
        self.assertEqual(bot.DEFAULT_SETTINGS["voice"], "Charon")
        self.assertFalse(bot.DEFAULT_SETTINGS["deep"])
        default_filter = bot.build_voice_filter({})
        self.assertNotIn("asetrate", default_filter)
        # The Deep toggle still exists for users who explicitly want it.
        self.assertIn("asetrate", bot.build_voice_filter({"deep": True}))

    async def test_tts_uses_charon_voice_and_neutral_prompt(self):
        captured = {}

        def fake_generate(model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(data=b"x" * 4000)
                                )
                            ]
                        )
                    )
                ]
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=fake_generate)
        )
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                    SpeechConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                    VoiceConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                    PrebuiltVoiceConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                ),
            )
        )

        def fake_run_cmd(cmd, timeout=600):
            bot.np.zeros(4800, dtype=bot.np.int16).tofile(cmd[-1])
            return SimpleNamespace(returncode=0)

        self.start_patch(patch.object(bot, "run_cmd", new=fake_run_cmd))
        self.start_patch(
            patch.object(
                bot,
                "split_pcm_by_weights",
                new=lambda master, batch: [
                    bot.np.zeros(2400, dtype=bot.np.float32) for _ in batch
                ],
            )
        )

        clips = await bot._tts_one_batch(
            [{"burmese": "ကျွန်တော် အိမ်ပြန်ခဲ့တယ်။"}],
            0,
            1,
            str(self.tmp_path),
            {},
        )

        self.assertIsNotNone(clips)
        voice_name = (
            captured["config"]
            .speech_config.voice_config.prebuilt_voice_config.voice_name
        )
        self.assertEqual(voice_name, "Charon")
        self.assertIn("ကျွန်တော် အိမ်ပြန်ခဲ့တယ်။", captured["contents"])
        # Nothing may instruct the voice to sound deep or slow — the
        # user asked for the original Charon delivery.
        self.assertNotIn("deep", captured["contents"])
        self.assertNotIn("အလွန်မြန်မြန်", captured["contents"])

    def _patch_gemini_json_types(self):
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        self.start_patch(
            patch.object(bot.asyncio, "sleep", new=AsyncMock(return_value=None))
        )

    async def test_direct_gemini_json_waits_out_overload_round_then_succeeds(self):
        # Regression: real job died with 503 UNAVAILABLE "high demand"
        # on every model because all attempts were burned within
        # seconds.  Overload failures must trigger extra backed-off
        # rounds so a temporary spike can clear.
        updates = []

        async def fake_update(text, force=False):
            updates.append(text)

        tracker = SimpleNamespace(update=fake_update)
        payload = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "ကျွန်တော်"}]}
        calls = []

        def flaky(model, contents, config):
            calls.append(model)
            if len(calls) <= 2:
                raise RuntimeError(
                    "503 UNAVAILABLE. {'error': {'code': 503, 'message': "
                    "'This model is currently experiencing high demand.', "
                    "'status': 'UNAVAILABLE'}}"
                )
            return SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=flaky)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self._patch_gemini_json_types()

        data = await bot._generate_direct_gemini_json(
            fake_client, ["contents"], purpose="test script", tracker=tracker
        )

        self.assertEqual(data, payload)
        # Round 1: m1, m2 both 503 -> backed-off round 2: m1 succeeds.
        self.assertEqual(calls, ["m1", "m2", "m1"])
        self.assertTrue(
            any("အလုပ်များနေပါသည်" in text for text in updates),
            updates,
        )

    async def test_direct_gemini_json_overload_rounds_are_bounded(self):
        calls = []

        def always_overloaded(model, contents, config):
            calls.append(model)
            raise RuntimeError("503 UNAVAILABLE high demand")

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=always_overloaded)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self.start_patch(patch.object(bot, "GEMINI_OVERLOAD_MAX_ROUNDS", 3))
        self._patch_gemini_json_types()

        with self.assertRaises(RuntimeError) as ctx:
            await bot._generate_direct_gemini_json(
                fake_client, ["contents"], purpose="test script", tracker=None
            )

        self.assertIn("failed after 6 attempts", str(ctx.exception))
        self.assertEqual(calls, ["m1", "m2"] * 3)

    async def test_direct_gemini_json_hard_error_fails_after_one_round(self):
        calls = []

        def hard_fail(model, contents, config):
            calls.append(model)
            raise RuntimeError("400 INVALID_ARGUMENT")

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=hard_fail)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self._patch_gemini_json_types()

        with self.assertRaises(RuntimeError) as ctx:
            await bot._generate_direct_gemini_json(
                fake_client, ["contents"], purpose="test script", tracker=None
            )

        self.assertIn("failed after 2 attempts", str(ctx.exception))
        self.assertEqual(calls, ["m1", "m2"])

    async def test_direct_recap_skips_retry_prompt_when_initial_call_fails(self):
        # If the initial script *call* itself failed on every model,
        # the second full retry-prompt call would repeat the identical
        # long wait.  It must be skipped so the job fails fast instead.
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/test-video",
        )
        calls = []

        def always_fail(model, contents, config):
            calls.append(model)
            raise RuntimeError("400 INVALID_ARGUMENT")

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=always_fail),
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self.start_patch(
            patch.object(
                bot,
                "types",
                SimpleNamespace(
                    GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
            )
        )
        self.start_patch(
            patch.object(bot.asyncio, "sleep", new=AsyncMock(return_value=None))
        )
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        with self.assertRaises(bot.GeminiScriptError):
            await bot.build_chunk_recap(
                str(self.tmp_path / "chunk.mp4"),
                100,
                0,
                100,
                {},
                1,
                tracker,
                settings={"coverage": 50, "rate": 10.0},
            )
        # Only the initial call ran (one pass over the models); the
        # retry prompt would have doubled this to 4 calls.
        self.assertEqual(calls, ["m1", "m2"])

    # --- v5.17: script recovery (production "no usable Burmese scenes") ---
    def test_scenes_from_plain_narration_builds_timed_burmese_scenes(self):
        prose = (
            "ကျွန်တော် မနက်စာ စားနေတုန်း တံခါးခေါက်သံ ကြားလိုက်တယ်။ "
            "ထွက်ကြည့်တော့ အိမ်နီးချင်း က စာအိတ်တစ်လုံး ပေးခဲ့တယ်။ "
            "အထဲမှာ ကျွန်တော် မမျှော်လင့်ထားတဲ့ စာတစ်စောင် ပါနေတယ်။ "
            "အဲဒီစာက ကျွန်တော့်ဘဝကို လုံးဝ ပြောင်းလဲပစ်လိုက်တယ်။"
        )

        scenes = bot._scenes_from_plain_narration(prose, 60.0, 4)

        self.assertEqual(len(scenes), 4)
        joined = " ".join(scene["burmese"] for scene in scenes)
        for sentence in prose.split("။"):
            sentence = sentence.strip()
            if sentence:
                self.assertIn(sentence, joined)
        self.assertEqual(scenes[0]["start"], 0.0)
        self.assertAlmostEqual(scenes[-1]["end"], 60.0, places=2)
        for scene in scenes:
            self.assertGreaterEqual(scene["start"], 0.0)
            self.assertLessEqual(scene["end"], 60.0)
            self.assertGreaterEqual(scene["end"], scene["start"])
            self.assertGreaterEqual(bot._burmese_ratio(scene["burmese"]), 0.35)
        for previous, current in zip(scenes, scenes[1:]):
            self.assertLessEqual(previous["end"], current["start"] + 0.001)

    def test_scenes_from_plain_narration_rejects_non_burmese_text(self):
        english = (
            "I woke up early and made some coffee before the meeting. "
            "Then I walked to the office and talked to my team about it."
        )
        self.assertEqual(bot._scenes_from_plain_narration(english, 60.0, 3), [])
        self.assertEqual(bot._scenes_from_plain_narration("", 60.0, 3), [])
        self.assertEqual(
            bot._scenes_from_plain_narration("ကျွန်တော်", 60.0, 3), []
        )

    def test_prose_from_direct_payload_finds_longest_burmese_string(self):
        payload = {
            "protagonist": "ကိုအောင်",
            "note": "short",
            "summary": (
                "ကျွန်တော် မနက်စာ စားနေတုန်း တံခါးခေါက်သံ ကြားလိုက်တယ်။ "
                "ထွက်ကြည့်တော့ အိမ်နီးချင်း က စာအိတ်တစ်လုံး ပေးခဲ့တယ်။"
            ),
        }
        prose = bot._prose_from_direct_payload(payload)
        self.assertTrue(prose.startswith("ကျွန်တော်"))
        self.assertEqual(bot._prose_from_direct_payload({"a": 1}), "")

    async def test_direct_gemini_json_plain_text_mode_returns_prose(self):
        prose = (
            "ကျွန်တော် မနက်စာ စားနေတုန်း တံခါးခေါက်သံ ကြားလိုက်တယ်။ "
            "ထွက်ကြည့်တော့ အိမ်နီးချင်း က စာအိတ်တစ်လုံး ပေးခဲ့တယ်။"
        )
        configs = []

        def gen(model, contents, config):
            configs.append(config)
            return SimpleNamespace(text=prose)

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1"]))
        self._patch_gemini_json_types()

        data = await bot._generate_direct_gemini_json(
            fake_client,
            ["contents"],
            purpose="test script plain text",
            tracker=None,
            plain_text=True,
        )

        self.assertEqual(data, prose)
        # No JSON envelope is requested: the whole point is to stop
        # fighting with response_mime_type when scenes come back empty.
        self.assertFalse(getattr(configs[0], "response_mime_type", ""))

    async def test_direct_recap_recovers_from_empty_json_scenes_with_plain_text(
        self,
    ):
        # Real failure: every model answered JSON, the payload parsed,
        # but it carried scene 0 -> the job died with "no usable
        # Burmese scenes".  The plain-text retelling must rescue it.
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/test-video",
        )
        prose = (
            "ကျွန်တော် မနက်စာ စားနေတုန်း တံခါးခေါက်သံ ကြားလိုက်တယ်။ "
            "ထွက်ကြည့်တော့ အိမ်နီးချင်း က စာအိတ်တစ်လုံး ပေးခဲ့တယ်။ "
            "အထဲမှာ ကျွန်တော် မမျှော်လင့်ထားတဲ့ စာတစ်စောင် ပါနေတယ်။"
        )
        mime_types = []

        def gen(model, contents, config):
            mime_types.append(getattr(config, "response_mime_type", ""))
            if getattr(config, "response_mime_type", ""):
                return SimpleNamespace(
                    text=json.dumps(
                        {"protagonist": "", "scenes": [], "cuts": []},
                        ensure_ascii=False,
                    )
                )
            return SimpleNamespace(text=prose)

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=gen),
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1"]))
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self._patch_gemini_json_types()
        self.start_patch(patch.object(bot, "SCRIPT_TOPUP_ATTEMPTS", 0))
        self.start_patch(
            patch.object(bot.asyncio, "sleep", new=AsyncMock(return_value=None))
        )
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))
        captured = []

        class _StopAfterScript(Exception):
            pass

        def stop(scenes):
            captured.extend(scenes)
            raise _StopAfterScript()

        self.start_patch(patch.object(bot, "sort_scenes", new=stop))
        self.addCleanup(setattr, bot, "_LAST_SCRIPT_RECOVERY", "")
        self.addCleanup(setattr, bot, "_LAST_SCRIPT_RAW_PREVIEW", "")
        self.addCleanup(setattr, bot, "_LAST_SCRIPT_ERROR", "")

        with self.assertRaises(_StopAfterScript):
            await bot.build_chunk_recap(
                str(self.tmp_path / "chunk.mp4"),
                60,
                0,
                60,
                {},
                1,
                tracker,
                settings={"coverage": 50, "rate": 10.0},
            )

        self.assertTrue(captured, "plain-text fallback produced no scenes")
        self.assertGreaterEqual(len(captured), 1)
        self.assertTrue(
            all(bot._burmese_ratio(scene["burmese"]) >= 0.35 for scene in captured)
        )
        self.assertEqual(bot._LAST_SCRIPT_RECOVERY, "plain_text_retelling")
        self.assertIn("", mime_types)
        self.assertEqual(bot._LAST_SCRIPT_ERROR, "")

    async def test_direct_recap_error_shows_what_gemini_answered(self):
        # The terminal message must not blame quota when the models
        # actually answered: it has to carry the diagnostics and a
        # snippet of the response so the failure is debuggable.
        video_file = SimpleNamespace(
            state=SimpleNamespace(name="ACTIVE"),
            name="files/test-video",
        )

        def gen(model, contents, config):
            return SimpleNamespace(
                text=json.dumps(
                    {"protagonist": "ကိုအောင်", "scenes": [], "cuts": []},
                    ensure_ascii=False,
                )
            )

        fake_client = SimpleNamespace(
            files=SimpleNamespace(
                upload=lambda file: video_file,
                get=lambda name: video_file,
                delete=lambda name: None,
            ),
            models=SimpleNamespace(generate_content=gen),
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1"]))
        self.start_patch(
            patch.object(
                bot,
                "genai",
                SimpleNamespace(Client=lambda api_key: fake_client),
            )
        )
        self._patch_gemini_json_types()
        self.addCleanup(setattr, bot, "_LAST_SCRIPT_ERROR", "")
        self.addCleanup(setattr, bot, "_LAST_SCRIPT_RECOVERY", "")
        tracker = SimpleNamespace(update=AsyncMock(return_value=None))

        with self.assertRaises(bot.GeminiScriptError) as ctx:
            await bot.build_chunk_recap(
                str(self.tmp_path / "chunk.mp4"),
                60,
                0,
                60,
                {},
                1,
                tracker,
                settings={"coverage": 50, "rate": 10.0},
            )

        message = str(ctx.exception)
        self.assertIn("no usable", message)
        self.assertIn("plain text", message)
        self.assertNotIn("quota ပြည့်တာ", message)
        self.assertIn("ကိုအောင်", message)
        self.assertIn("scenes", bot._LAST_SCRIPT_RAW_PREVIEW)
        self.assertEqual(bot._LAST_SCRIPT_RECOVERY, "")
        self.assertIn("no usable", bot._LAST_SCRIPT_ERROR)

    async def test_synthesize_tts_runs_two_batches_concurrently_in_order(self):
        state = {"active": 0, "max": 0}

        async def fake_one_batch(batch, b_idx, total, work_dir, settings=None):
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            await asyncio.sleep(0.05)
            state["active"] -= 1
            return [
                bot.np.full(10, float(b_idx + 1), dtype=bot.np.float32)
                for _ in batch
            ]

        self.start_patch(patch.object(bot, "_tts_one_batch", new=fake_one_batch))
        scenes = [{"burmese": f"s{i}"} for i in range(20)]

        clips = await bot.synthesize_tts(scenes, str(self.tmp_path), {})

        self.assertEqual(len(clips), 20)
        self.assertEqual(
            [float(c[0]) for c in clips],
            [1.0] * 8 + [2.0] * 8 + [3.0] * 4,
        )
        self.assertEqual(state["max"], 2)

    async def test_tts_one_batch_falls_back_and_scans_all_parts(self):
        # Regression: TTS failed on every model in production (100%
        # silent scenes).  The batch must (a) fall through to the
        # next model, (b) find the audio even when it is not in
        # parts[0], and (c) remember the failing model's error for
        # diagnostics.
        batch = [{"burmese": "က"}, {"burmese": "ခ"}]
        audio = b"\x00\x01" * 4000
        good_resp = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(inline_data=None),
                            SimpleNamespace(
                                inline_data=SimpleNamespace(data=audio)
                            ),
                        ]
                    )
                )
            ]
        )

        def gen(model, contents, config):
            if model == "bad-tts":
                raise RuntimeError("400 INVALID_ARGUMENT: bad voice config")
            return good_resp

        def fake_run_cmd(cmd, timeout):
            with open(cmd[-1], "wb") as f:
                # >1s of samples so split_pcm_by_weights really splits.
                f.write(b"\x10\x00" * bot.SAMPLE_RATE * 2)
            return SimpleNamespace(returncode=0)

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(
            patch.object(
                bot, "PREFERRED_TTS_MODELS", ["bad-tts", "good-tts"]
            )
        )
        # This test exercises the Gemini path; pin provider order.
        self.start_patch(
            patch.object(bot, "TTS_PROVIDER_ORDER", ("gemini", "edge"))
        )
        self.start_patch(
            patch.object(bot, "_make_gemini_client", lambda: fake_client)
        )
        self.start_patch(patch.object(bot, "run_cmd", fake_run_cmd))
        edge_mock = AsyncMock()
        self.start_patch(
            patch.object(bot, "_edge_tts_synthesize_to_file", new=edge_mock)
        )
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")
        self.addCleanup(setattr, bot, "_LAST_TTS_PROVIDER", "")

        clips = await bot._tts_one_batch(
            batch, 0, 1, str(self.tmp_path), None
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(clips), 2)
        self.assertIn("bad-tts", bot._LAST_TTS_ERROR)
        self.assertIn("400", bot._LAST_TTS_ERROR)
        # Gemini worked -> fallback untouched, provider recorded.
        edge_mock.assert_not_called()
        self.assertEqual(bot._LAST_TTS_PROVIDER, "gemini")

    async def test_tts_one_batch_all_fail_records_error_for_gates(self):
        # When every TTS model fails, the batch returns None (silence
        # fallback) and the last error is recorded so the silent-
        # scene gate message can name the actual cause.
        def gen(model, contents, config):
            raise RuntimeError("503 UNAVAILABLE: high demand")

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(
            patch.object(bot, "PREFERRED_TTS_MODELS", ["only-tts"])
        )
        self.start_patch(
            patch.object(bot, "_make_gemini_client", lambda: fake_client)
        )
        # Edge fallback is tried after Gemini -- make it fail too so
        # the batch exercises the total-failure path (and no network).
        self.start_patch(
            patch.object(
                bot,
                "_edge_tts_synthesize_to_file",
                new=AsyncMock(side_effect=RuntimeError("edge down")),
            )
        )
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "only-tts", None)
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "edge-tts", None)

        clips = await bot._tts_one_batch(
            [{"burmese": "က"}], 0, 1, str(self.tmp_path), None
        )

        self.assertIsNone(clips)
        self.assertIn("only-tts", bot._LAST_TTS_ERROR)
        self.assertIn("last TTS error", bot._last_tts_error_suffix())
        self.assertIn("edge down", bot._LAST_TTS_ERROR_BY_MODEL["edge-tts"])

    async def test_tts_one_batch_remembers_quota_exhausted_model(self):
        # Production: gemini-2.5-flash-preview-tts answered 429
        # RESOURCE_EXHAUSTED "You exceeded your current quota".  The
        # model must be tried once (no pointless retries/sleeps),
        # remembered as quota-exhausted, skipped by later calls, and
        # its full error kept per-model for /healthz diagnostics.
        calls = []

        def gen(model, contents, config):
            calls.append(model)
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED. You exceeded your current "
                "quota, please check your plan and billing details."
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(
            patch.object(bot, "PREFERRED_TTS_MODELS", ["quota-tts"])
        )
        self.start_patch(
            patch.object(bot, "_make_gemini_client", lambda: fake_client)
        )
        # Keep the Edge fallback out of this test (and off network).
        self.start_patch(
            patch.object(
                bot,
                "_edge_tts_synthesize_to_file",
                new=AsyncMock(side_effect=RuntimeError("edge down")),
            )
        )
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(bot._QUOTA_EXHAUSTED_MODELS.pop, "quota-tts", None)
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "edge-tts", None)
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "quota-tts", None)
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")

        clips = await bot._tts_one_batch(
            [{"burmese": "က"}], 0, 1, str(self.tmp_path), None
        )

        self.assertIsNone(clips)
        self.assertEqual(calls, ["quota-tts"])  # no retry storm
        self.assertIn("quota-tts", bot._quota_exhausted_active())
        self.assertIn("429", bot._LAST_TTS_ERROR_BY_MODEL["quota-tts"])
        self.assertEqual(
            bot._live_gemini_models(["quota-tts", "fresh-model"]),
            ["fresh-model"],
        )

    async def test_tts_one_batch_accepts_wav_audio(self):
        # Current TTS models return a complete WAV file; ffmpeg must
        # sniff it instead of being told it is headerless s16le.
        wav_audio = b"RIFF" + b"\x00" * 40 + b"\x10\x00" * 4000
        resp = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                inline_data=SimpleNamespace(data=wav_audio)
                            )
                        ]
                    )
                )
            ]
        )
        seen_cmds = []

        def fake_run_cmd(cmd, timeout):
            seen_cmds.append(cmd)
            with open(cmd[-1], "wb") as f:
                f.write(b"\x10\x00" * bot.SAMPLE_RATE * 2)
            return SimpleNamespace(returncode=0)

        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda model, contents, config: resp
            )
        )
        self.start_patch(
            patch.object(bot, "PREFERRED_TTS_MODELS", ["wav-tts"])
        )
        self.start_patch(
            patch.object(bot, "TTS_PROVIDER_ORDER", ("gemini", "edge"))
        )
        self.start_patch(
            patch.object(bot, "_make_gemini_client", lambda: fake_client)
        )
        self.start_patch(patch.object(bot, "run_cmd", fake_run_cmd))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")

        clips = await bot._tts_one_batch(
            [{"burmese": "က"}], 0, 1, str(self.tmp_path), None
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(seen_cmds), 1)
        input_args = seen_cmds[0][: seen_cmds[0].index("-i")]
        self.assertNotIn("s16le", input_args)

    async def test_tts_one_batch_edge_fallback_when_gemini_quota_dead(self):
        # Production endgame: every Gemini TTS model quota-exhausted
        # (429) failed whole jobs at 100% silent scenes.  The batch
        # must now complete via the free Edge Burmese voice instead.
        batch = [{"burmese": "က"}, {"burmese": "ခ"}]
        edge_calls = []

        def gen(model, contents, config):
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED. You exceeded your current "
                "quota, please check your plan and billing details."
            )

        async def fake_edge_save(text, voice, out_path):
            edge_calls.append((text, voice))
            with open(out_path, "wb") as f:
                f.write(b"\xff\xfb" + b"\x00" * 4000)  # fake MP3 bytes

        def fake_run_cmd(cmd, timeout):
            with open(cmd[-1], "wb") as f:
                f.write(b"\x00\x40" * bot.SAMPLE_RATE * 2)
            return SimpleNamespace(returncode=0)

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(
            patch.object(bot, "PREFERRED_TTS_MODELS", ["quota-tts-e1"])
        )
        # Premise: Gemini is tried first and fails, Edge rescues.
        self.start_patch(
            patch.object(bot, "TTS_PROVIDER_ORDER", ("gemini", "edge"))
        )
        self.start_patch(
            patch.object(bot, "_make_gemini_client", lambda: fake_client)
        )
        self.start_patch(
            patch.object(bot, "_edge_tts_synthesize_to_file", new=fake_edge_save)
        )
        self.start_patch(patch.object(bot, "run_cmd", fake_run_cmd))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(bot._QUOTA_EXHAUSTED_MODELS.pop, "quota-tts-e1", None)
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "quota-tts-e1", None)
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")
        self.addCleanup(setattr, bot, "_LAST_TTS_PROVIDER", "")

        clips = await bot._tts_one_batch(
            batch, 0, 1, str(self.tmp_path), None
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(clips), 2)
        self.assertEqual(bot._LAST_TTS_PROVIDER, "edge")
        # Per-scene synthesis: one Edge request per scene, each with
        # exactly that scene's text (no joined slicing guesses).
        self.assertEqual(len(edge_calls), 2)
        self.assertEqual([c[0] for c in edge_calls], ["က", "ခ"])
        # Default voice is Charon (male pronoun) -> male Burmese voice.
        self.assertEqual(edge_calls[0][1], "my-MM-ThihaNeural")
        self.assertIn("quota-tts-e1", bot._quota_exhausted_active())

    async def test_tts_edge_partial_scene_failure_keeps_good_scenes(self):
        # One flaky scene must not sink the batch: it becomes a
        # short silence, the good scenes keep their real audio.
        batch = [{"burmese": "က"}, {"burmese": "ခ"}]

        async def fake_edge_save(text, voice, out_path):
            if text == "ခ":
                raise RuntimeError("edge scene boom")
            with open(out_path, "wb") as f:
                f.write(b"\xff\xfb" + b"\x00" * 4000)

        def fake_run_cmd(cmd, timeout):
            with open(cmd[-1], "wb") as f:
                f.write(b"\x00\x40" * bot.SAMPLE_RATE)  # loud samples
            return SimpleNamespace(returncode=0)

        self.start_patch(
            patch.object(bot, "TTS_PROVIDER_ORDER", ("edge",))
        )
        self.start_patch(
            patch.object(bot, "_edge_tts_synthesize_to_file", new=fake_edge_save)
        )
        self.start_patch(patch.object(bot, "run_cmd", fake_run_cmd))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "edge-tts", None)
        self.addCleanup(setattr, bot, "_LAST_TTS_PROVIDER", "")

        clips = await bot._tts_one_batch(
            batch, 0, 1, str(self.tmp_path), None
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(clips), 2)
        self.assertEqual(bot._LAST_TTS_PROVIDER, "edge")
        self.assertFalse(bot._clip_is_silent(clips[0]))
        self.assertTrue(bot._clip_is_silent(clips[1]))
        self.assertIn("edge-tts", bot._LAST_TTS_ERROR_BY_MODEL)

    # --- v5.16: Edge TTS resilience (production "3/5 silent scenes") ---
    def _edge_text_clip_harness(self, fail_times=0, fail_always=False):
        """Fake Edge save/master that returns audio sized to the text.

        The audio for a text is ``len(text) / 2.3`` seconds of speech,
        i.e. what the real free Edge Burmese voice produces, so the clip
        validation bounds are exercised for real.
        """
        calls = []
        sleeps = []

        async def fake_save(text, voice, out_path):
            calls.append((text, voice))
            if fail_always or len(calls) <= fail_times:
                raise RuntimeError("Cannot connect to host speech.platform.bing.com")
            with open(out_path, "wb") as f:
                # >400 bytes: Edge's "produced no audio file" check.
                f.write(b"EDGE" + text.encode("utf-8") + b"\x00" * 600)

        async def fake_master(tmp_raw, tmp_pcm, voice_filter, input_args):
            with open(tmp_raw, "rb") as f:
                payload = f.read()[4:].split(b"\x00", 1)[0]
            text = payload.decode("utf-8")
            return np.full(
                max(1, int(bot.SAMPLE_RATE * len(text) / 2.3)),
                0.5,
                dtype=np.float32,
            )

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        self.start_patch(
            patch.object(bot, "_edge_tts_synthesize_to_file", new=fake_save)
        )
        self.start_patch(patch.object(bot, "_audio_file_to_master", new=fake_master))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=fake_sleep))
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "edge-tts", None)
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")
        self.addCleanup(setattr, bot, "_LAST_TTS_PROVIDER", "")
        self.addCleanup(setattr, bot, "_LAST_TTS_SILENCED_SCENES", 0)
        return calls, sleeps

    async def test_edge_scene_retries_transient_failures_with_backoff(self):
        text = MALE_PRONOUN + " မနက်စောစော အိမ်ကနေ ထွက်လာခဲ့တယ်။"
        calls, sleeps = self._edge_text_clip_harness(fail_times=2)

        clips = await bot._tts_one_batch_edge(
            [{"burmese": text}], 0, 1, str(self.tmp_path),
            dict(bot.DEFAULT_SETTINGS), "", text,
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(calls), 3)  # two transient failures, then OK
        self.assertEqual(sleeps[:2], [0.6, 1.5])
        self.assertFalse(bot._clip_is_silent(clips[0]))
        self.assertEqual(bot._LAST_TTS_SILENCED_SCENES, 0)
        self.assertEqual(bot._LAST_TTS_PROVIDER, "edge")

    async def test_edge_scene_retries_sentence_by_sentence_before_silence(self):
        # A single long request that keeps failing must not lose the
        # scene: re-request it sentence by sentence and rejoin the audio.
        first = MALE_PRONOUN + " မနက်စောစော အိမ်ကနေ ထွက်လာခဲ့တယ်။"
        second = "လမ်းမှာ မိုးက သည်းသည်းမည်းမည်း ရွာနေတယ်။"
        text = first + " " + second
        calls = []
        sleeps = []

        async def fake_save(text_arg, voice, out_path):
            calls.append(text_arg)
            if text_arg.strip() == text.strip():
                raise RuntimeError("edge ws closed")
            with open(out_path, "wb") as f:
                f.write(b"EDGE" + text_arg.encode("utf-8") + b"\x00" * 600)

        async def fake_master(tmp_raw, tmp_pcm, voice_filter, input_args):
            with open(tmp_raw, "rb") as f:
                payload = f.read()[4:].split(b"\x00", 1)[0].decode("utf-8")
            return np.full(
                max(1, int(bot.SAMPLE_RATE * len(payload) / 2.3)),
                0.5,
                dtype=np.float32,
            )

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        self.start_patch(
            patch.object(bot, "_edge_tts_synthesize_to_file", new=fake_save)
        )
        self.start_patch(patch.object(bot, "_audio_file_to_master", new=fake_master))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=fake_sleep))
        self.addCleanup(bot._LAST_TTS_ERROR_BY_MODEL.pop, "edge-tts", None)
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")
        self.addCleanup(setattr, bot, "_LAST_TTS_SILENCED_SCENES", 0)

        clips = await bot._tts_one_batch_edge(
            [{"burmese": text}], 0, 1, str(self.tmp_path),
            dict(bot.DEFAULT_SETTINGS), "", text,
        )

        self.assertIsNotNone(clips)
        self.assertFalse(bot._clip_is_silent(clips[0]))
        self.assertEqual(bot._LAST_TTS_SILENCED_SCENES, 0)
        # Whole-scene attempts all failed, then each sentence succeeded.
        sentence_calls = [c for c in calls if c.strip() in (first.strip(), second.strip())]
        self.assertEqual(len(sentence_calls), 2)
        expected = int(bot.SAMPLE_RATE * len(first) / 2.3) + int(
            bot.SAMPLE_RATE * len(second) / 2.3
        )
        self.assertAlmostEqual(clips[0].size, expected, delta=bot.SAMPLE_RATE // 2)

    async def test_edge_gives_up_early_when_the_free_voice_is_down(self):
        scenes = [
            {"burmese": MALE_PRONOUN + f" စာကြောင်း နံပါတ် {i} ဖြစ်တယ်။"}
            for i in range(5)
        ]
        calls, _sleeps = self._edge_text_clip_harness(fail_always=True)

        clips = await bot._tts_one_batch_edge(
            scenes, 0, 1, str(self.tmp_path),
            dict(bot.DEFAULT_SETTINGS), "", "",
        )

        self.assertIsNone(clips)  # next provider / retry pass takes over
        # Zero successes after two dead scenes -> stop instead of
        # grinding through every scene of the batch.
        self.assertEqual(
            len(calls),
            bot.EDGE_TTS_BATCH_ABORT_AFTER_FAILURES * bot.EDGE_TTS_SCENE_ATTEMPTS,
        )
        self.assertIn("edge-tts", bot._LAST_TTS_ERROR)
        self.assertIn("no success", bot._LAST_TTS_ERROR)
        self.assertIn("speech.platform.bing.com", bot._LAST_TTS_ERROR)

    async def test_synthesize_tts_recovers_silent_scenes_in_a_fresh_pass(self):
        calls = []

        async def fake_one_batch(batch, b_idx, total, work_dir, settings=None):
            calls.append((b_idx, len(batch)))
            if len(calls) == 1:
                return [
                    np.zeros(100, dtype=np.float32) for _ in batch
                ]
            return [np.full(1000, 0.4, dtype=np.float32) for _ in batch]

        self.start_patch(patch.object(bot, "_tts_one_batch", new=fake_one_batch))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(setattr, bot, "_LAST_TTS_SILENCED_SCENES", 0)
        self.addCleanup(setattr, bot, "_LAST_TTS_RETRY_RECOVERED", 0)

        scenes = [{"burmese": "က"}, {"burmese": "ခ"}]
        clips = await bot.synthesize_tts(scenes, str(self.tmp_path), {})

        self.assertEqual(len(clips), 2)
        self.assertEqual(len(calls), 2)  # first pass + silent-scene retry pass
        self.assertEqual(calls[1][1], 2)
        self.assertTrue(all(not bot._clip_is_silent(clip) for clip in clips))
        self.assertEqual(bot._LAST_TTS_SILENCED_SCENES, 0)
        self.assertEqual(bot._LAST_TTS_RETRY_RECOVERED, 2)

    def test_edge_tts_has_a_short_timeout_and_bounded_batch_abort(self):
        # A hung Edge WebSocket must not eat the 300s Gemini TTS timeout.
        self.assertGreaterEqual(bot.EDGE_TTS_TIMEOUT_SECONDS, 10)
        self.assertLessEqual(bot.EDGE_TTS_TIMEOUT_SECONDS, 120)
        self.assertEqual(bot.EDGE_TTS_SCENE_ATTEMPTS, 4)
        self.assertEqual(bot.EDGE_TTS_BATCH_ABORT_AFTER_FAILURES, 2)
        self.assertEqual(len(bot.EDGE_TTS_RETRY_DELAYS), bot.EDGE_TTS_SCENE_ATTEMPTS - 1)

    def test_silent_scene_error_names_the_real_cause_in_burmese(self):
        self.addCleanup(setattr, bot, "_LAST_TTS_ERROR", "")
        bot._LAST_TTS_ERROR = (
            "edge-tts: 3/5 scene(s) silent (my-MM-ThihaNeural): edge ws closed"
        )
        message = bot._silent_scenes_error(3, 5)

        self.assertIn("အသံသွင်းမှု မအောင်မြင်ပါ", message)
        self.assertIn("3/5", message)
        self.assertIn("edge ws closed", message)
        # The free Edge voice is the working provider: never blame Gemini.
        self.assertNotIn("Gemini", message)

    def test_polish_clip_trims_dead_air_and_fades_edges(self):
        # Per-scene MP3s carry service/encoder silence at both ends;
        # joined raw, the narration went stop-start every few
        # seconds.  Polish must keep the speech (plus small pads),
        # drop the dead air, and make joins click-free.
        sr = bot.SAMPLE_RATE
        burst = (
            0.5 * bot.np.sin(2 * bot.np.pi * 180 * bot.np.arange(sr) / sr)
        ).astype(bot.np.float32)
        clip = bot.np.concatenate([
            bot.np.zeros(int(0.4 * sr), dtype=bot.np.float32),
            burst,
            bot.np.zeros(int(0.5 * sr), dtype=bot.np.float32),
        ])
        polished = bot._polish_clip(clip)
        self.assertGreater(polished.size, int(1.05 * sr))
        self.assertLess(polished.size, int(1.20 * sr))
        self.assertAlmostEqual(
            float(bot.np.max(bot.np.abs(polished))), 0.5, delta=0.02
        )
        self.assertAlmostEqual(float(polished[0]), 0.0, delta=0.01)
        self.assertAlmostEqual(float(polished[-1]), 0.0, delta=0.01)
        # Explicit silence fallbacks pass through untouched so the
        # duration/silence gates still see them.
        silence = bot.np.zeros(int(2.5 * sr), dtype=bot.np.float32)
        self.assertEqual(bot._polish_clip(silence).size, silence.size)

    def test_voice_filter_loudnorm_lives_on_master_only(self):
        # Per-clip loudnorm normalized every few-second scene on its
        # own -> loudness jumped between scenes.  It must run once,
        # on the assembled master narration.
        self.assertNotIn("loudnorm", bot.build_voice_filter({}))
        self.assertNotIn("loudnorm", bot.build_voice_filter({"deep": True}))
        self.assertIn("asetrate", bot.build_voice_filter({"deep": True}))
        self.assertIn("atempo=1.500", bot.build_voice_filter({"speed": 1.5}))
        self.assertNotIn("volume", bot.build_voice_filter({"volume": 50}))
        self.assertIn("loudnorm", bot.build_master_voice_filter({}))
        self.assertIn(
            "volume=0.50", bot.build_master_voice_filter({"volume": 50})
        )

    async def test_tts_edge_retries_truncated_scene_clip(self):
        # A decodable-but-truncated Edge result (stream ended
        # mid-word) used to pass the old >=400-byte check and was
        # heard as cut-off speech.  It must be rejected and retried.
        batch = [{"burmese": "က" * 80}]  # ~12s at the planning rate
        decode_calls = []

        async def fake_edge_save(text, voice, out_path):
            with open(out_path, "wb") as f:
                f.write(b"\xff\xfb" + b"\x00" * 4000)

        def fake_run_cmd(cmd, timeout):
            decode_calls.append(cmd)
            seconds = 0.5 if len(decode_calls) == 1 else 10
            with open(cmd[-1], "wb") as f:
                f.write(b"\x00\x40" * int(bot.SAMPLE_RATE * seconds))
            return SimpleNamespace(returncode=0)

        self.start_patch(
            patch.object(bot, "TTS_PROVIDER_ORDER", ("edge",))
        )
        self.start_patch(
            patch.object(bot, "_edge_tts_synthesize_to_file", new=fake_edge_save)
        )
        self.start_patch(patch.object(bot, "run_cmd", fake_run_cmd))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(setattr, bot, "_LAST_TTS_PROVIDER", "")

        clips = await bot._tts_one_batch(
            batch, 0, 1, str(self.tmp_path), None
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(decode_calls), 2)  # truncated -> retried
        self.assertFalse(bot._clip_is_silent(clips[0]))
        self.assertGreater(clips[0].size, bot.SAMPLE_RATE * 5)

    async def test_tts_default_order_is_free_edge_first(self):
        # User rule: completely free, no queueing.  With the default
        # provider order the Edge voice must win immediately and
        # Gemini (quota queue) must not even be called.
        self.assertEqual(bot.TTS_PROVIDER_ORDER[0], "edge")
        batch = [{"burmese": "က"}, {"burmese": "ခ"}]
        gemini_calls = []

        def gen(model, contents, config):
            gemini_calls.append(model)
            raise RuntimeError("should not be called")

        async def fake_edge_save(text, voice, out_path):
            with open(out_path, "wb") as f:
                f.write(b"\xff\xfb" + b"\x00" * 4000)

        def fake_run_cmd(cmd, timeout):
            with open(cmd[-1], "wb") as f:
                f.write(b"\x00\x40" * bot.SAMPLE_RATE * 2)
            return SimpleNamespace(returncode=0)

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(
            patch.object(bot, "_make_gemini_client", lambda: fake_client)
        )
        self.start_patch(
            patch.object(bot, "_edge_tts_synthesize_to_file", new=fake_edge_save)
        )
        self.start_patch(patch.object(bot, "run_cmd", fake_run_cmd))
        self.start_patch(patch.object(bot.asyncio, "sleep", new=AsyncMock()))
        self.addCleanup(setattr, bot, "_LAST_TTS_PROVIDER", "")

        clips = await bot._tts_one_batch(
            batch, 0, 1, str(self.tmp_path), None
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(clips), 2)
        self.assertEqual(bot._LAST_TTS_PROVIDER, "edge")
        self.assertEqual(gemini_calls, [])

    def test_edge_tts_voice_is_always_thiha(self):
        # User rule: the Edge narration voice is always Thiha,
        # whatever Gemini voice/pronoun is selected.
        for voice in ("Charon", "Kore", "Aoede"):
            self.assertEqual(
                bot._edge_tts_voice_for({"voice": voice}),
                "my-MM-ThihaNeural",
            )

    def test_calibration_trim_uses_effective_rate_not_inflated_spoken_rate(self):
        # Production regression: narration measured 120.0s vs a
        # 113.0s target (acceptable 110.5-115.5), yet calibration
        # refused to render.  Some clips measured as silent, so the
        # spoken-only chars/second was inflated and the old trim
        # math produced a desired char total ABOVE the current
        # script -- the trim was a no-op and calibration stalled.
        # The desired total must come from the effective rate (all
        # chars over all produced speech seconds).
        measurements = {
            "speech_seconds": 118.0,
            "pause_total_seconds": 1.8,
            # Inflated spoken-only rate: 2000 chars over the 100s of
            # non-silent slices--the value that used to break trims.
            "measured_chars_per_second": 20.0,
        }
        # Old formula would have asked for ~2203 chars (> current).
        trim_to = bot._calibration_desired_total_chars(
            measurements, 2000, 113.0, expand=False
        )
        self.assertEqual(
            trim_to, int((113.0 - 1.8) * (2000 / 118.0) * 0.99)
        )
        self.assertLess(trim_to, 2000)

        expand_to = bot._calibration_desired_total_chars(
            measurements, 2000, 113.0, expand=True
        )
        self.assertEqual(
            expand_to, int((113.0 - 1.8) * (2000 / 118.0) * 1.01) + 1
        )
        self.assertGreater(expand_to, trim_to)

        # Without speech measurements the helper keeps current size.
        self.assertEqual(
            bot._calibration_desired_total_chars({}, 1500, 113.0, expand=False),
            1500,
        )

    def test_minimum_script_chars_allows_near_complete_script(self):
        # Production regression: a complete first-person script of
        # 715 chars for a 734-char estimate was refused for being
        # 5 chars under the old 98% minimum -- before measured-TTS
        # calibration (which enforces the real duration and can
        # expand) ever ran.  The char gate is only an anti-summary
        # floor now.
        self.assertLessEqual(bot._minimum_script_chars(734), 715)
        # A half-length summary is still refused.
        self.assertGreater(bot._minimum_script_chars(734), 367)

    def test_tts_speech_config_shapes(self):
        new_cfg = bot._tts_speech_config("Charon", "voice")
        self.assertEqual(new_cfg.voice_config.voice, "Charon")
        old_cfg = bot._tts_speech_config("Charon", "prebuilt")
        self.assertEqual(
            old_cfg.voice_config.prebuilt_voice_config.voice_name, "Charon"
        )

    async def test_direct_gemini_json_hedges_when_first_model_is_slow(self):
        # Regression: one model was observed processing a script call
        # for 100+ seconds while other models sat idle, because
        # failover only happened after the full per-attempt timeout.
        # Hedging must start the next model in parallel and take the
        # first successful answer.
        payload_slow = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "slow"}]}
        payload_fast = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "fast"}]}
        calls = []

        def gen(model, contents, config):
            calls.append(model)
            if model == "m1":
                time.sleep(1.2)
                return SimpleNamespace(
                    text=json.dumps(payload_slow, ensure_ascii=False)
                )
            return SimpleNamespace(
                text=json.dumps(payload_fast, ensure_ascii=False)
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self.start_patch(patch.object(bot, "GEMINI_HEDGE_DELAY_SECONDS", 0.2))
        self.start_patch(
            patch.object(bot, "GEMINI_GENERATE_TIMEOUT_SECONDS", 5)
        )
        self._patch_gemini_json_types()

        started = time.monotonic()
        data = await bot._generate_direct_gemini_json(
            fake_client, ["contents"], purpose="test script", tracker=None
        )
        elapsed = time.monotonic() - started

        self.assertEqual(data, payload_fast)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(calls[0], "m1")
        self.assertIn("m2", calls)

    async def test_direct_gemini_json_races_models_immediately_for_video(self):
        # Fastest-answer mode: when the call carries the video, the
        # first models start *together* -- the user must not wait for
        # a slow first model while others sit idle.  The hedge delay
        # is patched huge so only the initial race can explain the
        # fast model winning immediately.
        payload_slow = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "slow"}]}
        payload_fast = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "fast"}]}
        calls = []

        def gen(model, contents, config):
            calls.append(model)
            if model == "m1":
                time.sleep(1.2)
                return SimpleNamespace(
                    text=json.dumps(payload_slow, ensure_ascii=False)
                )
            return SimpleNamespace(
                text=json.dumps(payload_fast, ensure_ascii=False)
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self.start_patch(patch.object(bot, "GEMINI_HEDGE_DELAY_SECONDS", 60))
        self.start_patch(
            patch.object(bot, "GEMINI_HEDGE_INITIAL_PARALLEL", 2)
        )
        self.start_patch(
            patch.object(bot, "GEMINI_GENERATE_TIMEOUT_SECONDS", 5)
        )
        self._patch_gemini_json_types()

        started = time.monotonic()
        data = await bot._generate_direct_gemini_json(
            fake_client,
            [SimpleNamespace(name="fake-uploaded-video"), "prompt"],
            purpose="test script",
            tracker=None,
        )
        elapsed = time.monotonic() - started

        self.assertEqual(data, payload_fast)
        self.assertLess(elapsed, 1.0)
        self.assertIn("m2", calls)

    async def test_direct_gemini_json_text_only_call_does_not_race(self):
        # Text-only calls (repairs/summaries) stay quota-lean: one
        # model at a time, hedged only after the delay.  With the
        # hedge delay patched huge, the slow first model must win
        # and the second must never launch.
        payload_slow = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "slow"}]}
        calls = []

        def gen(model, contents, config):
            calls.append(model)
            time.sleep(0.3)
            return SimpleNamespace(
                text=json.dumps(payload_slow, ensure_ascii=False)
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(patch.object(bot, "PREFERRED_MODELS", ["m1", "m2"]))
        self.start_patch(patch.object(bot, "GEMINI_HEDGE_DELAY_SECONDS", 60))
        self.start_patch(
            patch.object(bot, "GEMINI_HEDGE_INITIAL_PARALLEL", 2)
        )
        self.start_patch(
            patch.object(bot, "GEMINI_GENERATE_TIMEOUT_SECONDS", 5)
        )
        self._patch_gemini_json_types()

        data = await bot._generate_direct_gemini_json(
            fake_client, ["prompt only"], purpose="test script", tracker=None
        )

        self.assertEqual(data, payload_slow)
        self.assertEqual(calls, ["m1"])

    async def test_direct_gemini_json_remembers_unavailable_models(self):
        # Regression: Google retired gemini-2.5-flash-lite for this API
        # key (404 NOT_FOUND "no longer available"), yet every window
        # kept launching it and its 404 became the reported error.  A
        # model that 404s must be remembered and never launched again
        # in this process.
        payload = {"scenes": [{"start": 0.0, "end": 5.0, "burmese": "x"}]}
        calls = []

        def gen(model, contents, config):
            calls.append(model)
            if model == "gone-model-t1":
                raise RuntimeError(
                    "404 NOT_FOUND. This model is no longer available"
                )
            return SimpleNamespace(
                text=json.dumps(payload, ensure_ascii=False)
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(
            patch.object(
                bot, "PREFERRED_MODELS", ["gone-model-t1", "ok-model-t1"]
            )
        )
        self._patch_gemini_json_types()
        self.addCleanup(bot._DEAD_GEMINI_MODELS.discard, "gone-model-t1")

        first = await bot._generate_direct_gemini_json(
            fake_client, ["c"], purpose="test script", tracker=None
        )
        self.assertEqual(first, payload)
        self.assertIn("gone-model-t1", bot._DEAD_GEMINI_MODELS)

        calls.clear()
        second = await bot._generate_direct_gemini_json(
            fake_client, ["c"], purpose="test script", tracker=None
        )
        self.assertEqual(second, payload)
        # The dead model is not even launched on later calls.
        self.assertEqual(calls, ["ok-model-t1"])

    async def test_direct_gemini_json_overload_error_not_masked_by_404(self):
        # When flash models are overloaded (503) and a trailing model
        # is retired (404), the final error must report the overload
        # (the real, transient cause), and later overload rounds must
        # not relaunch the retired model.
        calls = []

        def gen(model, contents, config):
            calls.append(model)
            if model == "gone-model-t2":
                raise RuntimeError("404 NOT_FOUND")
            raise RuntimeError(
                "503 UNAVAILABLE. This model is currently experiencing "
                "high demand"
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=gen)
        )
        self.start_patch(
            patch.object(
                bot, "PREFERRED_MODELS", ["busy-model-t2", "gone-model-t2"]
            )
        )
        self.start_patch(patch.object(bot, "GEMINI_OVERLOAD_MAX_ROUNDS", 2))
        self._patch_gemini_json_types()
        self.addCleanup(bot._DEAD_GEMINI_MODELS.discard, "gone-model-t2")

        with self.assertRaises(RuntimeError) as ctx:
            await bot._generate_direct_gemini_json(
                fake_client, ["c"], purpose="test script", tracker=None
            )

        self.assertIn("503", str(ctx.exception))
        self.assertNotIn("404", str(ctx.exception))
        # Round 1: busy + gone.  Round 2: busy only (gone remembered).
        self.assertEqual(
            calls, ["busy-model-t2", "gone-model-t2", "busy-model-t2"]
        )


if __name__ == "__main__":
    unittest.main()

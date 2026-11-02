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

        def fake_extract(_source, output_path, _start, _duration):
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
        self.assertEqual((target, low, high), (540.0, 528.0, 552.0))
        self.assertEqual(bot.script_target_chars(100, settings), 1800)
        self.assertTrue(bot._duration_is_on_target(540, 600, settings))
        self.assertFalse(bot._duration_is_on_target(33, 600, settings))
        self.assertFalse(bot._duration_is_on_target(599, 600, settings))
        self.assertEqual(bot._duration_status(33, 600, settings), "short")
        self.assertEqual(bot._duration_status(599, 600, settings), "long")

        scenes = [
            {"group_id": 1, "start": 0, "burmese": first_person_text(100)},
            {"group_id": 2, "start": 10, "burmese": first_person_text(100)},
        ]
        trimmed = bot.trim_scenes_proportionally(scenes, 100)
        self.assertEqual(bot.script_total_chars(trimmed), 100)
        self.assertTrue(
            all(
                bot._scene_is_first_person(scene["burmese"], MALE_PRONOUN)
                for scene in trimmed
            )
        )

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
            prompt = str(contents[1]) if len(contents) > 1 else ""
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
        self.assertIn("FIRST-PERSON REPAIR ONLY", generate_calls[1])
        self.assertIn("ယခင် scenes များကို ပြန်မပေးပါနှင့်", generate_calls[2])
        final_duration, _, _ = bot.probe_video_info(out_chunk)
        self.assertAlmostEqual(final_duration, 2.75, delta=0.2)

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
        self.assertIn("could not be calibrated", str(raised.exception))
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
        self.assertAlmostEqual(audio["total_narration_seconds"], 90.0, delta=0.05)
        self.assertEqual(script["coverage"]["actual_burmese_chars"], 1800)
        self.assertGreaterEqual(len(audio["duration_calibration_history"]), 3)
        self.assertEqual(audio["duration_calibration_history"][0]["narration_seconds"], 45.0)
        self.assertGreaterEqual(len(tts_calls), 3)
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
            self.assertEqual(health.json()["minimum_script_coverage_percent"], 90)
            self.assertEqual(health.json()["target_final_duration_percent"], 90)
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
                "inline_persistent_menu_and_per_part_recap",
            )
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
        # This is the RECAP_FLOW=auto compatibility path; the default
        # manual button flow has dedicated tests below.
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
        ):
            build_calls.append(
                {
                    "path": chunk_path,
                    "duration": actual_dur,
                    "start": start,
                    "end": end,
                    "context": dict(story_context),
                    "part": part_num,
                    "settings": dict(settings),
                }
            )
            output = Path(chunk_path).parents[1] / f"recap_part_{part_num:04d}.mp4"
            output.write_bytes(b"r" * 100)
            return str(output), {
                "characters": ["Hero", "Mina"],
                "summary": f"after part {part_num}",
                "last_scene": f"scene ending {part_num}",
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
            }

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
        self.assertEqual([call["part"] for call in build_calls], [1, 2, 3])
        self.assertEqual(
            [(call["start"], call["end"]) for call in build_calls],
            [(0.0, 120.0), (120.0, 240.0), (240.0, 250.0)],
        )
        self.assertTrue(
            all(call["settings"]["chunk"] == 120 for call in build_calls)
        )
        self.assertEqual(build_calls[1]["context"]["summary"], "after part 1")
        self.assertEqual(build_calls[2]["context"]["summary"], "after part 2")
        self.assertEqual(build_calls[2]["context"]["protagonist"], "Hero")
        self.assertEqual(build_calls[2]["context"]["characters"], ["Hero", "Mina"])
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

        async def fake_build(chunk_path, *_args):
            output = Path(chunk_path).parent / "recap_part_0001.mp4"
            output.write_bytes(b"r" * 100)
            return str(output), {
                "characters": ["Hero"],
                "summary": "done",
                "last_scene": "end",
                "protagonist": "Hero",
                "pronoun": MALE_PRONOUN,
            }

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

    async def test_n8n_forwarding_failure_falls_back_to_direct_pipeline(self):
        self.start_patch(patch.object(bot, "MANUAL_PARTS_ENABLED", False))
        self.start_patch(patch.object(bot, "PIPELINE_MODE", "n8n"))
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


if __name__ == "__main__":
    unittest.main()

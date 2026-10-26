import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
        start(patch.object(bot, "N8N_WEBHOOK_URL", ""))
        start(patch.object(bot, "N8N_WEBHOOK_SECRET", ""))

        self.active_jobs = start(patch.object(bot, "ACTIVE_JOBS", {}))
        self.worker_tasks = start(patch.object(bot, "WORKER_TASKS", {}))
        self.cancelled_jobs = start(patch.object(bot, "CANCELLED_JOBS", set()))
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
        tasks = [task for task in bot.WORKER_TASKS.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        bot.WORKER_TASKS.clear()
        bot.ACTIVE_JOBS.clear()
        bot.CANCELLED_JOBS.clear()
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
                    {"group_id": 2, "start": 10, "end": 15, "burmese": "second"},
                    {"group_id": 1, "start": 0, "end": 5, "burmese": "first"},
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
                        "burmese": "scene",
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

        await bot._worker_stage_render(state)

        final = job_dir / "final.mp4"
        self.assertTrue(final.exists() and final.stat().st_size > 0)
        final_duration, _, _ = bot.probe_video_info(str(final))
        self.assertAlmostEqual(final_duration, 0.4, delta=0.15)
        plan = json.loads((job_dir / "render_plan.json").read_text(encoding="utf-8"))["items"]
        self.assertEqual(plan[0]["desired_output_seconds"], 0.4)
        self.assertEqual(plan[0]["input_seconds"], 0.4)
        self.assertEqual(state["metrics"]["render"]["audio_alignment_filters_used"], [])

    async def test_worker_api_auth_and_health(self):
        async with self.api_client() as client:
            health = await client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["version"], "v3.1-n8n-worker")

            root = await client.get("/")
            self.assertEqual(root.status_code, 200)
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

    async def test_n8n_forwarding_failure_falls_back_to_direct_pipeline(self):
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
        self.start_patch(patch.object(bot, "get_source_duration", new=direct_entry))

        await bot.handle_link_job(14001, 55, "https://videos.example/fallback")

        forward.assert_awaited_once()
        forwarded_state = forward.await_args.args[0]
        self.assertEqual(forwarded_state["job_id"], "jabc123def0")
        self.assertFalse((self.jobs_dir / "jabc123def0").exists())
        direct_entry.assert_awaited_once_with("https://videos.example/fallback")
        self.assertNotIn(14001, bot.ACTIVE_JOBS)


if __name__ == "__main__":
    unittest.main()

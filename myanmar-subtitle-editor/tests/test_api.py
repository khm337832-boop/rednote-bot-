"""API စမ်းသပ်မှုများ — chunk upload, transcribe job, project CRUD, export, SRT, auth."""
from __future__ import annotations

import dataclasses
import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import config as cfg
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def uploaded_video(client, sample_video):
    size = sample_video.stat().st_size
    start = client.post(
        "/api/upload/start", json={"filename": "sample.mp4", "size": size}
    )
    assert start.status_code == 200, start.text
    upload_id = start.json()["upload_id"]
    offset = 0
    with open(sample_video, "rb") as handle:
        while True:
            chunk = handle.read(512 * 1024)
            if not chunk:
                break
            response = client.put(
                f"/api/upload/{upload_id}?offset={offset}",
                content=chunk,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            if payload.get("resync"):
                offset = payload["received"]
                handle.seek(offset)
                continue
            offset += len(chunk)
    done = client.post(f"/api/upload/{upload_id}/complete")
    assert done.status_code == 200, done.text
    return done.json()


def wait_for_job(client: TestClient, job_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"done", "error", "cancelled"}:
            return last
        time.sleep(0.15)
    raise AssertionError(f"အလုပ် အချိန်ကုန်သွားသည်: {last}")


def test_capabilities_reports_ffmpeg_and_whisper(client):
    data = client.get("/api/capabilities").json()
    assert data["ffmpeg"]["ok"] is True
    assert data["ffmpeg"]["has_libass"] is True
    assert data["ffmpeg"]["has_harfbuzz"] is True
    assert data["fonts"]["family"] == "Noto Sans Myanmar"
    assert data["fonts"]["regular"] is True
    assert data["config"]["max_upload_mb"] >= 100
    assert "whisper" in data
    assert data["limits"]["max_upload_mb"] == data["config"]["max_upload_mb"]


def test_healthz(client):
    data = client.get("/healthz").json()
    assert data["status"] == "ok"


def test_upload_rejects_bad_extension(client):
    response = client.post("/api/upload/start", json={"filename": "virus.exe", "size": 100})
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client):
    too_big = cfg.get_settings().max_upload_bytes + 1
    response = client.post(
        "/api/upload/start", json={"filename": "big.mp4", "size": too_big}
    )
    assert response.status_code == 413


def test_video_metadata_and_range_stream(client, uploaded_video):
    meta = client.get(f"/api/video/{uploaded_video['id']}/meta").json()
    assert meta["width"] == 1280 and meta["height"] == 720
    assert 2.5 <= meta["duration"] <= 3.6

    full = client.get(f"/api/video/{uploaded_video['id']}")
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"

    partial = client.get(
        f"/api/video/{uploaded_video['id']}", headers={"Range": "bytes=0-99"}
    )
    assert partial.status_code == 206
    assert len(partial.content) == 100
    assert partial.headers["content-range"].startswith("bytes 0-99/")


def test_transcribe_demo_job(client, uploaded_video):
    response = client.post(
        "/api/transcribe", json={"video_id": uploaded_video["id"], "demo": True}
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "done", job
    assert job["progress"] == pytest.approx(1.0)
    segments = job["result"]["segments"]
    assert len(segments) >= 1
    for current, following in zip(segments, segments[1:]):
        assert current["end"] == pytest.approx(following["start"], abs=1e-3)


def test_project_crud_and_export(client, uploaded_video, tmp_path):
    transcribe = client.post(
        "/api/transcribe", json={"video_id": uploaded_video["id"], "demo": True}
    )
    job = wait_for_job(client, transcribe.json()["job_id"])
    segments = [
        {
            "id": f"s{index}",
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
            "style": {"fontSize": 0.06, "y": 0.82, "width": 0.8},
        }
        for index, item in enumerate(job["result"]["segments"])
    ]

    # logo upload
    buffer = io.BytesIO()
    Image.new("RGBA", (240, 120), (255, 80, 160, 220)).save(buffer, format="PNG")
    buffer.seek(0)
    logo_response = client.post(
        "/api/logo", files={"file": ("logo.png", buffer, "image/png")}
    )
    assert logo_response.status_code == 200, logo_response.text
    logo = logo_response.json()
    assert logo["src"].startswith("logos/")
    assert client.get(logo["url"]).status_code == 200

    project_payload = {
        "name": "စမ်းသပ် ပရောဂျက်",
        "video": {
            "id": uploaded_video["id"],
            "filename": uploaded_video["filename"],
            "width": uploaded_video["width"],
            "height": uploaded_video["height"],
            "duration": uploaded_video["duration"],
        },
        "segments": segments,
        "logos": [
            {
                "id": "logoone",
                "src": logo["src"],
                "x": 0.05,
                "y": 0.05,
                "w": 0.2,
                "h": 0.1,
                "rotation": 10,
                "opacity": 0.9,
            }
        ],
        "blurs": [
            {"id": "blurone", "x": 0.6, "y": 0.6, "w": 0.2, "h": 0.2, "intensity": 14}
        ],
    }
    saved = client.post("/api/project/save", json=project_payload)
    assert saved.status_code == 200, saved.text
    project = saved.json()["project"]
    assert project["id"]
    assert project["warnings"] == []
    for current, following in zip(project["segments"], project["segments"][1:]):
        assert current["end"] == pytest.approx(following["start"], abs=1e-3)

    listed = client.get("/api/project/list").json()["projects"]
    assert any(item["id"] == project["id"] for item in listed)

    fetched = client.get(f"/api/project/{project['id']}").json()["project"]
    assert fetched["name"] == "စမ်းသပ် ပရောဂျက်"
    assert len(fetched["segments"]) == len(segments)

    # SRT (UTF-8 BOM)
    srt = client.post("/api/srt", json={"project": project})
    assert srt.status_code == 200
    assert srt.content.startswith(b"\xef\xbb\xbf")
    assert "မင်္ဂလာပါ" in srt.content.decode("utf-8-sig")
    assert "attachment" in srt.headers["content-disposition"]

    # Export
    export = client.post("/api/export", json={"project": project})
    assert export.status_code == 200, export.text
    export_job = wait_for_job(client, export.json()["job_id"])
    assert export_job["status"] == "done", export_job
    download_url = export_job["result"]["download_url"]
    rendered = client.get(download_url)
    assert rendered.status_code == 200
    assert len(rendered.content) > 5000
    assert export_job["result"]["size"] == len(rendered.content)

    # SRT from the saved project
    srt_saved = client.get(f"/api/srt/{project['id']}")
    assert srt_saved.status_code == 200

    deleted = client.delete(f"/api/project/{project['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/api/project/{project['id']}").status_code == 404


def test_project_save_normalises_overlaps(client):
    payload = {
        "name": "ထပ်နေသော စာတန်း",
        "video": {"id": "", "duration": 10},
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "ပထမ"},
            {"start": 2.0, "end": 5.0, "text": "ဒုတိယ"},
            {"start": 9.0, "end": 5.0, "text": "ပျက်"},
        ],
    }
    project = client.post("/api/project/save", json=payload).json()["project"]
    assert project["warnings"]
    assert len(project["segments"]) == 2
    assert project["segments"][0]["end"] == pytest.approx(project["segments"][1]["start"])
    client.delete(f"/api/project/{project['id']}")


def test_export_requires_segments(client, uploaded_video):
    response = client.post(
        "/api/export",
        json={
            "project": {
                "name": "ဗလာ",
                "video": {"id": uploaded_video["id"]},
                "segments": [],
            }
        },
    )
    assert response.status_code == 400


def test_missing_job_returns_404(client):
    assert client.get("/api/jobs/doesnotexist1234").status_code == 404


def test_password_protection(monkeypatch):
    settings = cfg.get_settings()
    protected = dataclasses.replace(settings, access_password="let-me-in")
    monkeypatch.setattr(cfg, "_settings", protected)
    with TestClient(app) as guarded:
        assert guarded.get("/api/project/list").status_code == 401
        assert guarded.get("/api/auth/status").json() == {"required": True, "authed": False}
        bad = guarded.post("/api/auth/login", json={"password": "wrong"})
        assert bad.status_code == 401
        ok = guarded.post("/api/auth/login", json={"password": "let-me-in"})
        assert ok.status_code == 200
        assert guarded.get("/api/project/list").status_code == 200
        # pages stay reachable so the login overlay can render
        assert guarded.get("/").status_code == 200


def test_transcribe_job_translates_and_keeps_timing(client, uploaded_video, monkeypatch):
    from backend import main
    monkeypatch.setattr(main.cfg, 'whisper_available', lambda settings: (True, ''))
    monkeypatch.setattr(main.sg, 'transcribe_video', lambda *a, **k: {
        'language': 'en', 'segments': [
            {'start': 0.2, 'end': 1.6, 'speech_end': 1.0, 'text': 'hello'},
            {'start': 1.6, 'end': 2.9, 'speech_end': 2.0, 'text': 'today'},
        ], 'words': [], 'duration': 3.0,
    })
    monkeypatch.setattr(main.tr, 'translate_segments', lambda items, lang: [
        {**item, 'text': text} for item, text in zip(items, ['မင်္ဂလာပါ', 'ဒီနေ့'])
    ])
    result = client.post('/api/transcribe', json={'video_id': uploaded_video['id']})
    assert result.status_code == 200
    job = wait_for_job(client, result.json()['job_id'])
    assert job['status'] == 'done'
    assert job['result']['translated_to'] == 'my'
    assert [(s['start'], s['end']) for s in job['result']['segments']] == [(0.2, 1.6), (1.6, 2.9)]
    assert job['result']['segments'][0]['text'] == 'မင်္ဂလာပါ'


def test_render_deployment_without_private_password_fails_closed(client, monkeypatch):
    monkeypatch.setenv('RENDER', 'true')
    monkeypatch.setattr(cfg.get_settings(), 'access_password', '')
    assert client.get('/api/project/list').status_code == 503
    assert client.get('/api/auth/status').status_code == 503
    assert client.get('/healthz').status_code == 200

# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient

from app.api.main import api
from tests.conftest import make_image_bytes, TOP3_STUB, TOP1_STUB
from herbs_detection.model_registry import ENABLED_KEYS


@pytest.fixture
def client():
    with TestClient(api) as c:
        yield c


class TestRoot:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_lists_enabled_models(self, client):
        assert set(client.get("/").json()["models"]) == set(ENABLED_KEYS)


class TestListModels:
    def test_returns_all_enabled(self, client):
        data = client.get("/models").json()
        assert len(data) == len(ENABLED_KEYS)
        assert all("key" in m and "img_size" in m for m in data)


class TestPredict:
    def _post(self, client, models="all", top_k=3, suffix=".jpg"):
        img = make_image_bytes(suffix)
        return client.post(
            "/predict",
            files={"file": (f"plant{suffix}", img, "image/jpeg")},
            data={"models": models, "top_k": top_k},
        )

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_all_models_returned_by_default(self, client):
        returned = {p["model"] for p in self._post(client).json()["predictions"]}
        assert returned == set(ENABLED_KEYS)

    def test_single_model_selection(self, client):
        data = self._post(client, models="convnext_tiny").json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["model"] == "convnext_tiny"

    def test_multiple_model_selection(self, client):
        returned = {p["model"] for p in
                    self._post(client, models="convnext_tiny,resnet50").json()["predictions"]}
        assert returned == {"convnext_tiny", "resnet50"}

    def test_top3_default(self, client):
        assert len(self._post(client).json()["predictions"][0]["top3"]) == 3

    def test_top_k_one(self, client):
        assert len(self._post(client, top_k=1).json()["predictions"][0]["top3"]) == 1

    def test_prediction_shape(self, client):
        pred = self._post(client).json()["predictions"][0]
        assert "model" in pred and "top3" in pred
        assert "class" in pred["top3"][0] and "confidence" in pred["top3"][0]

    def test_unknown_model_returns_422(self, client):
        assert self._post(client, models="does_not_exist").status_code == 422

    def test_accepts_png(self, client):
        assert self._post(client, suffix=".png").status_code == 200


class TestPredictBatch:
    def _post(self, client, n=2, models="all"):
        img = make_image_bytes()
        files = [("files", (f"img{i}.jpg", img, "image/jpeg")) for i in range(n)]
        return client.post("/predict-batch", files=files, data={"models": models})

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_one_result_per_image(self, client):
        assert len(self._post(client, n=3).json()) == 3

    def test_result_has_filename(self, client):
        assert self._post(client, n=1).json()[0]["filename"] == "img0.jpg"

    def test_all_models_present(self, client):
        returned = {p["model"] for p in self._post(client, n=1).json()[0]["predictions"]}
        assert returned == set(ENABLED_KEYS)

    def test_model_selection(self, client):
        result = self._post(client, n=1, models="convnext_tiny").json()[0]
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["model"] == "convnext_tiny"


class TestExplore:
    def _post(self, client, top_k=3):
        img = make_image_bytes()
        return client.post(
            "/explore",
            files={"file": ("plant.jpg", img, "image/jpeg")},
            data={"top_k": top_k},
        )

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_contains_all_models(self, client):
        returned = {p["model"] for p in self._post(client).json()["predictions"]}
        assert returned == set(ENABLED_KEYS)

    def test_filename_in_response(self, client):
        assert self._post(client).json()["filename"] == "plant.jpg"

    def test_each_entry_has_rank(self, client):
        entries = self._post(client).json()["predictions"][0]["top_k"]
        assert entries[0]["rank"] == 1
        assert entries[1]["rank"] == 2


def test_get_metrics_returns_expected_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "kpis" in data
    assert "recent_requests" in data
    assert "class_distribution" in data
    assert "model_stats" in data


def test_get_metrics_predictions_returns_list(client):
    resp = client.get("/metrics/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)

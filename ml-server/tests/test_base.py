import glob
def test_base(app_client):
    response = app_client.get("/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("Welcome to the image classifier")


def test_classify_healthy(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("images/healthy.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0]["label"] == "healthy"
    assert response.json()["data"][0]["score"] > 0.9


def test_classify_early_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("images/early_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0]["label"] == "early_blight"
    assert response.json()["data"][0]["score"] > 0.9


def test_classify_late_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("images/late_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0]["label"] == "late_blight"
    assert response.json()["data"][0]["score"] > 0.9
    
def test_batch(app_client):
    files = [('files', open(i, 'rb')) for i in glob.glob('images/*.jpeg')]
    labels = [i.split('/')[1].split('.')[0] for i in glob.glob('images/*.jpeg')]
    
    response = app_client.post(
        "/classify_batch",
        files=files,
    )
    resp_labels = [i[0]["label"] for i in response.json()["data"]]
    resp_scores = [i[0]["score"] for i in response.json()["data"]]
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert len(resp_labels) == len(labels)
    assert all([a == b for a, b in zip(resp_labels, labels)])
    assert all(i >= 0.9 for i in resp_scores)
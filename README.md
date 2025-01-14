# ML Take Home

Here are some words about my solution to test task. 

* But at first unzip `dataset/potato_plans_diseases.zip` to `dataset/PLD_3_Classes_256`
* Load `pytorch_model.bin` and place it to `ml-server/mlmodels/vit-potatoes-plant-health-status/pytorch_model.bin`. You can download model from here https://drive.google.com/file/d/1n_bw7KF09DcwsgMu4YK99vSeQHJPhCyf/view?usp=share_link

### Mandatory Tasks

1. As it was recommended I used `google/vit-base-patch16-224-in21k` model. The code for training is located in `train.py`, but also I put here a jupyter notebook `classifier_transformers.ipynb`. 
2. Besides there is another jupyter notebook `classifier_lightning.ipynb`. It contains code to train another model (`resnet50`). I provided it to show that I am familiar with the `pytorch-lightning` framework. I also think that lightning is more configurable than transformers. But for other tasks I used  `ViT` model.
3. I put finetuned model in the `mlmodels` folder and updated `classifier.py`. Also all tests are passed.
4. For training I used [SaturnCloud](https://saturncloud.io/).

### Optional tasks
1. I added `classify_batch` method in `app.py` to analyze multiple images at once. Also added batch test at `tests/test_base.py`.
2. I dockerized the application. In fact, there are two containers `ml-client` and `ml-server`. `Dockerfile` for each of them is located in the corresponding folder. To run the whole application I used the docker-compose and `docker-compose.yaml` is located in the root folder. I commented lines that correspond to nginx, because it's only for cloud deployment. In fact, you just need to run `docker compose up -d`. Also I pushed containers to the DockerHub, so you can find them by `markpotanin/ml-client` and `markpotanin/ml-server`. 
3. I deployed the application in a cloud provider (I used [Linode](https://www.linode.com/)). You can view the web page at `http://<SERVERIP>:3000/`.
4. I added two additional features. I thought - we are developing real application for users that can classify plants/flowers diseases. It will be userful not just classify the disease, but also provide some advices how to cure your lovely plant. 
Just this week OpenAI released API acess to ChatGPT, and provided trial period to try it. So I added `chatgpt` method to `app.py`. It just send a request to openai API using my API token, no rocket science. But I just wanted to try it :-)
Usage example could be found in `notebooks/requests.ipynb`. In this notebook I send some requests to running model to check if it works.

5. The second additional feature, that would be userful - Image Similarity. So we upload a new photo and classify it. And then want to find similar images from our dataset to check the classification. or to show the user plants with similar problems. In short - we compute embeddings of images and then use the cosine similarity to determine how similar the two images are. The code to calculate embeddings of train dataset is located in `notebooks/similarity.ipynb`. So `app.py` will use file with precomputed embeddings, which is located in `ml-server/mlmodels/file.npz`. I could use libraries like FAISS or ScanNN, or optimize this similarity pipeline, but I think it's beyond the scope of this task.



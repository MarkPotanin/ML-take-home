class CFG:
    save_path = "./mlmodels/vit-potatoes-plant-health-status"
    train_embeddings_path = "./mlmodels/train_embeddings.npz"
    base_model = "google/vit-base-patch16-224-in21k"
    data_path = "../dataset/PLD_3_Classes_256/"
    batch_size = 32
    epohcs = 4
    fp16 = True
    learning_rate = 2e-4
class DefaultCofig(object):

    # log info
    log_path = './logs/'

    # model info
    model = 'RefineNet'
    batch_size = 1
    use_gpu = True
    epochs = 800
    lr = 1e-6
    momentum = 0.9
    decay = 1e-4
    original_lr = 1e-6
    steps = [-1, 1, 100, 150]
    scales = [1, 1, 1, 1]
    workers = 4

    # train&test files path
    images = 'data/nyu_images'
    labels = 'data/nyu_labels40'
    train_split = 'data/train.txt'
    test_split = 'data/test.txt'

    # saved model path
    saved_model_path = 'saved_models/'

    # test model file
    test_model = saved_model_path + 'RefineNet_1220_20:38:52.pkl'

    # predict files path
    predict_images = 'data/predict/images'
    predict_labels = 'data/predict/labels'
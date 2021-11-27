from torchreid.utils import load_pretrained_weights
from torchreid.utils import RankLogger
import torchreid.models
if __name__ == '__main__':

    model = torchreid.models.osnet_x1_0()
    

    weight_path = 'log/osnet_x1_0/model/model.pth.tar-5'
    load_pretrained_weights(model, weight_path=weight_path)
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )
    engine.run(
        max_epoch=20,
        eval_freq=10,
        print_freq=10,
        test_only=True,

    )

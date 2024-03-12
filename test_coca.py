import yaml
from coca import CoCa
from modules.encoders.tokenizer import BpeTokenizer
from PIL import Image

import mindspore as ms
from mindspore import ops, load_checkpoint, load_param_into_net
from mindspore.dataset import vision


def create_model(config, checkpoint_path = None):
    with open(
        config,
        "r",
    ) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # print(model_cfg)
    model = CoCa(**model_cfg)
    if checkpoint_path is not None:
        checkpoint_param = load_checkpoint(checkpoint_path)
        load_param_into_net(model, checkpoint_param)

    return model


if __name__ == "__main__":
    config = "/disk1/mindone/songyuanwei/mindone/examples/coca/coca_ViT-L-14.yaml"
    ms.set_context(mode=ms.PYNATIVE_MODE)

    im = Image.open("/disk1/mindone/songyuanwei/mindone/examples/coca/cat.jpg").convert("RGB")
    im = vision.ToTensor()(im)
    im = ms.Tensor(im).expand_dims(0)
    im = ops.ResizeBilinearV2()(im, (224, 224))
    # im = im.squeeze()
    print(im.shape)
    model = create_model(config)
    # with open("coca_ms_weight.text", "w", ) as f:
    #     for param in model.get_parameters():
    #         name = param.name
    #         value = param.data.asnumpy()
    #         f.write(f"{name}: {value.shape} \n")
    #         # print(name, value.shape)

    model.set_train(False)
    text = model.generate(im)
    tokenizer = BpeTokenizer()
    # output = tokenizer.decode(text[0].asnumpy()).split("<endoftext>")[0].replace("<startoftext>", "")
    output = tokenizer.decode(text[0].asnumpy())
    print(output)

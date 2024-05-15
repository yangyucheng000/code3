import mindspore as ms
from trm.config import cfg
from trm.modeling import build_model
import torch
import argparse
# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')['model']
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        if name == 'featpool.conv.weight':
            parameter = parameter.unsqueeze(3)
        if 'output.dense' in name or 'intermediate.dense' in name:
            print('**',name, parameter.numpy().shape)
            parameter = parameter.t()
        # if  "key"  in name or 'query' in name or 'value' in name:
        #     parameter = parameter.t()
        #     print('##',name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params

# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        # print(name, value.shape)
        ms_params[name] = value
    return ms_params




def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
            "beta": "bias",
            "moving_mean": "running_mean",
            "moving_variance": "running_var"}
    
    """

text_encoder.bert.encoder.layer.0.attention.output.dense.weight (768, 768)
text_encoder.bert.encoder.layer.0.attention.output.dense.bias (768,)






text_encoder.bert.bert_encoder.encoder.blocks.0.attention.projection.weight (768, 768)
text_encoder.bert.bert_encoder.encoder.blocks.0.attention.projection.bias (768,)



    """
    
    ms2pt = {"text_encoder.bert.word_embedding.embedding_table": "text_encoder.bert.embeddings.word_embeddings.weight",
                "text_encoder.bert.embedding_postprocessor.token_type_embedding.embedding_table": "text_encoder.bert.embeddings.token_type_embeddings.weight",
                "text_encoder.bert.embedding_postprocessor.full_position_embedding.embedding_table": "text_encoder.bert.embeddings.position_embeddings.weight",
                "text_encoder.bert.embedding_postprocessor.layernorm.gamma": "text_encoder.bert.embeddings.LayerNorm.weight",
                "text_encoder.bert.embedding_postprocessor.layernorm.beta": "text_encoder.bert.embeddings.LayerNorm.bias",
                "text_encoder.bert.dense.weight": "text_encoder.bert.pooler.dense.weight",
                "text_encoder.bert.dense.bias": "text_encoder.bert.pooler.dense.bias",
                "text_encoder.layernorm.gamma": "text_encoder.layernorm.weight",
                "text_encoder.layernorm.beta": "text_encoder.layernorm.bias",
                }
    layer_map = {# layer0
                "text_encoder.bert.bert_encoder.encoder.blocks.#.layernorm1.gamma": "text_encoder.bert.encoder.layer.#.output.LayerNorm.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.layernorm1.beta": "text_encoder.bert.encoder.layer.#.output.LayerNorm.bias",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.layernorm2.gamma": "text_encoder.bert.encoder.layer.#.attention.output.LayerNorm.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.layernorm2.beta": "text_encoder.bert.encoder.layer.#.attention.output.LayerNorm.bias",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.dense1.weight": "text_encoder.bert.encoder.layer.#.attention.self.query.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.dense1.bias": "text_encoder.bert.encoder.layer.#.attention.self.query.bias",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.dense2.weight": "text_encoder.bert.encoder.layer.#.attention.self.key.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.dense2.bias": "text_encoder.bert.encoder.layer.#.attention.self.key.bias",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.dense3.weight": "text_encoder.bert.encoder.layer.#.attention.self.value.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.dense3.bias": "text_encoder.bert.encoder.layer.#.attention.self.value.bias",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.output.mapping.weight": "text_encoder.bert.encoder.layer.#.intermediate.dense.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.output.mapping.bias": "text_encoder.bert.encoder.layer.#.intermediate.dense.bias",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.output.projection.weight": "text_encoder.bert.encoder.layer.#.output.dense.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.output.projection.bias": "text_encoder.bert.encoder.layer.#.output.dense.bias",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.projection.weight": "text_encoder.bert.encoder.layer.#.attention.output.dense.weight",
                "text_encoder.bert.bert_encoder.encoder.blocks.#.attention.projection.bias": "text_encoder.bert.encoder.layer.#.attention.output.dense.bias"}
    for i in range(12):
        for k, v in layer_map.items():
            ms2pt[k.replace('#', str(i))] = v.replace('#', str(i))
    new_params_list = []
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "bn" in ms_param or "downsample.1" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表

            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            elif ms2pt.get(ms_param,'') in pt_params and pt_params[ms2pt[ms_param]].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms2pt[ms_param]]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)



def main():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = num_gpus > 1

    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://"
    #     )
    #     synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(cfg)
    pth_path = "/hd1/shared/TRM_pytorch/outputs/charades/pool_model_9e.pth"
    pt_param = pytorch_params(pth_path)
    print("="*20)
    ckpt_path = "trm_charades_e9_all.ckpt"
    ms_param = mindspore_params(model)
    param_convert(ms_param, pt_param, ckpt_path)

if __name__ == "__main__":
    #mp.set_start_method('spawn')
    #
    main()
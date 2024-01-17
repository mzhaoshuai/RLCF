# coding=utf-8
from .coco_cap import COCOCLIPCapTrainDataset, COCOCapDatasetForEmbedding, Flickr30kCapDatasetForEmbedding, NocapsCapDatasetForEmbedding
from .coco_lmdb import COCOCLIPCapTrainDatasetLMDB


def get_dataset(args):
    # create dataset
    if not args.clip_patch:
        dataset = COCOCLIPCapTrainDataset(args.data, args.prefix_length, normalize_prefix=args.normalize_prefix,
                                            use_image_embedding=args.use_image_embedding,
                                            config_dir=args.llm_config_dir,
                                            force_gen_tokens=False
                                            )
    else:
        dataset = COCOCLIPCapTrainDatasetLMDB(args.data, args.prefix_length, normalize_prefix=args.normalize_prefix,
                                                use_image_embedding=args.use_image_embedding,
                                                config_dir=args.llm_config_dir,
                                                annotations=args.annotations
                                                )

    return dataset

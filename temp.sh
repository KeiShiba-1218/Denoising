CUDA_VISIBLE_DEVICES=5 python -m train_codes.train3 --config config/config_axialunet_large.json
CUDA_VISIBLE_DEVICES=4 python -m train_codes.train3 --config config/config_axialunet_large.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.train --config config/config_unet2.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.train2 --config config/config_unet_gray.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.train2 --config config/config_unet_gray.json

python -m eval_codes.visualize_losses --config experiments/resunet_mse2/config_unet_gray.json

python -m eval_codes.generate_images --config experiments/resunet/config_unet_gray.json

#cd ../../axialunet2/ckpt/
#ls | grep -v -E '*500000.ckpt' | xargs rm -f
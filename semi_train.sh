python semi_train.py --pretrained_model_path ./save/[2024-03-08-0144]_New_pt_bs6Acc2n5_crop96_ItkRot30_lr2e-3/best/best_froc_mean_recall.pth --mixed_precision --val_mixed_precision --unlabeled_train_set ./data/client0_unlabeled_train.txt --train_set ./data/client0_labeled_train.txt --val_set ./data/client0_val.txt --test_set ./data/all_client_test.txt --batch_size 6 --unlabeled_batch_size 6
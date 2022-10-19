1. Training: python3 main.py
2. Evaluation: python3 eval.py $1 $2
   •$1 the path of model's parameter
   •$2 the path of the annotation for test data
   •E.g. python3 eval.py --path ./save_dir/EfficientNet/best_model.pt --test_anno ./p2_data/annotations/public_test_annos.json

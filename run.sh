
#ds=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
#ds=("ETTh1" "ETTh2")
#ms=("Transformer" "FEDformer" "Autoformer" "Informer")
#vs=("Fourier" "Wavelets")
#for d in ${ds[@]}; do
#  for m in ${ms[@]}; do
#    for v in ${vs[@]}; do
#      echo "|||||||||||||||||||||||||||||"
#      python run.py --data ${d}  --model ${m} --version ${v}
#    done
#  done
#done
#

python run.py --data BTCUSDT_train --model FEDformer --version Fourier --data_path btcusdt_1m_2025_delta.csv --root_path ./dataset/BTC/ --features MS --target delta_close_price_ratio --seq_len 200 --label_len 100 --pred_len 1 --enc_in 9 --dec_in 9 --c_out 9 --use_multi_gpu --devices 1,2,3

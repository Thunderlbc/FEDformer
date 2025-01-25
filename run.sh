
#ds=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
ds=("ETTh1" "ETTh2")
ms=("Transformer" "FEDformer" "Autoformer" "Informer")
vs=("Fourier" "Wavelets")
for d in ${ds[@]}; do
  for m in ${ms[@]}; do
    for v in ${vs[@]}; do
      echo "|||||||||||||||||||||||||||||"
      python run.py --data ${d}  --model ${m} --version ${v}
    done
  done
done

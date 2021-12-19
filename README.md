# ASR project barebones

## Installation guide
```
git clone https://github.com/iamilyasedunov/asr_project_template.git
cd asr_project_template/bins
bash build_image.sh
bash run_container.sh
docker attach ISedunov-asr_template
cd /home/asr_project_template/asr_project_template
cd other/
bash load_files.sh
cd ../
python test.py -r other/model_best.pth -c other/config.json
```

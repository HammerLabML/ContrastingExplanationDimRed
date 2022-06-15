mkdir exp_results/

python experiments.py toy none 0 linear exp_results/ fancy > exp_results/toy_none_linear.txt
python experiments.py toy none 0 ae exp_results/ fancy > exp_results/toy_none_ae.txt
python experiments.py toy none 0 tsne exp_results/ fancy > exp_results/toy_none_tsne.txt
python experiments.py toy none 0 som exp_results/ fancy > exp_results/toy_none_som.txt

python experiments.py toy none 0 linear exp_results/ memory > exp_results/toy_none_linear_memory.txt
python experiments.py toy none 0 ae exp_results/ memory > exp_results/toy_none_ae_memory.txt
python experiments.py toy none 0 tsne exp_results/ memory > exp_results/toy_none_tsne_memory.txt
python experiments.py toy none 0 som exp_results/ memory > exp_results/toy_none_som_memory.txt

python experiments.py toy shift 3 linear exp_results/ fancy > exp_results/breastcancer_shift_linear_fancy.txt
python experiments.py toy shift 3 som exp_results/ fancy > exp_results/breastcancer_shift_som_fancy.txt
python experiments.py toy shift 3 ae exp_results/ fancy > exp_results/breastcancer_shift_ae_fancy.txt
python experiments.py toy shift 3 tsne exp_results/ fancy > exp_results/breastcancer_shift_tsne_fancy.txt
python experiments.py toy shift 3 linear exp_results/ memory > exp_results/diabetes_shift_linear_memory.txt
python experiments.py toy shift 3 som exp_results/ memory > exp_results/diabetes_shift_som_memory.txt
python experiments.py toy shift 3 ae exp_results/ memory > exp_results/diabetes_shift_ae_memory.txt
python experiments.py toy shift 3 tsne exp_results/ memory > exp_results/diabetes_shift_tsne_memory.txt

python experiments.py toy gaussian 3 linear exp_results/ fancy > exp_results/diabetes_gaussian_linear_fancy.txt
python experiments.py toy gaussian 3 som exp_results/ fancy > exp_results/diabetes_gaussian_som_fancy.txt
python experiments.py toy gaussian 3 ae exp_results/ fancy > exp_results/diabetes_gaussian_ae_fancy.txt
python experiments.py toy gaussian 3 tsne exp_results/ fancy > exp_results/diabetes_gaussian_tsne_fancy.txt
python experiments.py toy gaussian 3 linear exp_results/ memory > exp_results/breastcancer_gaussian_linear_memory.txt
python experiments.py toy gaussian 3 som exp_results/ memory > exp_results/breastcancer_gaussian_som_memory.txt
python experiments.py toy gaussian 3 ae exp_results/ memory > exp_results/breastcancer_gaussian_ae_memory.txt
python experiments.py toy gaussian 3 tsne exp_results/ memory > exp_results/breastcancer_gaussian_tsne_memory.txt

python experiments.py toy zero 3 linear exp_results/ fancy > exp_results/breastcancer_zero_linear_fancy.txt
python experiments.py toy zero 3 som exp_results/ fancy > exp_results/breastcancer_zero_som_fancy.txt
python experiments.py toy zero 3 ae exp_results/ fancy > exp_results/breastcancer_zero_ae_fancy.txt
python experiments.py toy zero 3 tsne exp_results/ fancy > exp_results/breastcancer_zero_tsne_fancy.txt
python experiments.py toy zero 3 linear exp_results/ memory > exp_results/diabetes_zero_linear_memory.txt
python experiments.py toy zero 3 som exp_results/ memory > exp_results/diabetes_zero_som_memory.txt
python experiments.py toy zero 3 ae exp_results/ memory > exp_results/diabetes_zero_ae_memory.txt
python experiments.py toy zero 3 tsne exp_results/ memory > exp_results/diabetes_zero_tsne_memory.txt


####################################################################################################################


python experiments.py breastcancer none 0 linear exp_results/ fancy > exp_results/breastcancer_none_linear.txt
python experiments.py breastcancer none 0 ae exp_results/ fancy > exp_results/breastcancer_none_ae.txt
python experiments.py breastcancer none 0 tsne exp_results/ fancy > exp_results/breastcancer_none_tsne.txt
python experiments.py breastcancer none 0 som exp_results/ fancy > exp_results/breastcancer_none_som.txt

python experiments.py diabetes none 0 linear exp_results/ fancy > exp_results/diabetes_none_linear.txt
python experiments.py diabetes none 0 ae exp_results/ fancy > exp_results/diabetes_none_ae.txt
python experiments.py diabetes none 0 tsne exp_results/ fancy > exp_results/diabetes_none_tsne.txt
python experiments.py diabetes none 0 som exp_results/ fancy > exp_results/diabetes_none_som.txt


python experiments.py breastcancer shift 3 linear exp_results/ fancy > exp_results/breastcancer_shift_linear.txt
python experiments.py breastcancer shift 3 ae exp_results/ fancy > exp_results/breastcancer_shift_ae.txt
python experiments.py breastcancer shift 3 tsne exp_results/ fancy > exp_results/breastcancer_shift_tsne.txt
python experiments.py breastcancer shift 3 som exp_results/ fancy > exp_results/breastcancer_shift_som.txt

python experiments.py diabetes shift 3 linear exp_results/ fancy > exp_results/diabetes_shift_linear.txt
python experiments.py diabetes shift 3 ae exp_results/ fancy > exp_results/diabetes_shift_ae.txt
python experiments.py diabetes shift 3 tsne exp_results/ fancy > exp_results/diabetes_shift_tsne.txt
python experiments.py diabetes shift 3 som exp_results/ fancy > exp_results/diabetes_shift_som.txt

python experiments.py diabetes gaussian 3 linear exp_results/ fancy > exp_results/diabetes_gaussian_linear.txt
python experiments.py diabetes gaussian 3 ae exp_results/ fancy > exp_results/diabetes_gaussian_ae.txt
python experiments.py diabetes gaussian 3 tsne exp_results/ fancy > exp_results/diabetes_gaussian_tsne.txt
python experiments.py diabetes gaussian 3 som exp_results/ fancy > exp_results/diabetes_gaussian_som.txt

python experiments.py breastcancer gaussian 3 linear exp_results/ fancy > exp_results/breastcancer_gaussian_linear.txt
python experiments.py breastcancer gaussian 3 ae exp_results/ fancy > exp_results/breastcancer_gaussian_ae.txt
python experiments.py breastcancer gaussian 3 tsne exp_results/ fancy > exp_results/breastcancer_gaussian_tsne.txt
python experiments.py breastcancer gaussian 3 som exp_results/ fancy > exp_results/breastcancer_gaussian_som.txt

python experiments.py breastcancer zero 3 linear exp_results/ fancy > exp_results/breastcancer_zero_linear.txt
python experiments.py breastcancer zero 3 ae exp_results/ fancy > exp_results/breastcancer_zero_ae.txt
python experiments.py breastcancer zero 3 tsne exp_results/ fancy > exp_results/breastcancer_zero_tsne.txt
python experiments.py breastcancer zero 3 som exp_results/ fancy > exp_results/breastcancer_zero_som.txt

python experiments.py diabetes zero 3 linear exp_results/ fancy > exp_results/diabetes_zero_linear.txt
python experiments.py diabetes zero 3 ae exp_results/ fancy > exp_results/diabetes_zero_ae.txt
python experiments.py diabetes zero 3 tsne exp_results/ fancy > exp_results/diabetes_zero_tsne.txt
python experiments.py diabetes zero 3 som exp_results/ fancy > exp_results/diabetes_zero_som.txt


##########################################################################################################################################

python experiments.py breastcancer none 0 linear exp_results/ memory > exp_results/breastcancer_none_linear_memory.txt
python experiments.py breastcancer none 0 ae exp_results/ memory > exp_results/breastcancer_none_ae_memory.txt
python experiments.py breastcancer none 0 tsne exp_results/ memory > exp_results/breastcancer_none_tsne_memory.txt
python experiments.py breastcancer none 0 som exp_results/ memory > exp_results/breastcancer_none_som_memory.txt

python experiments.py diabetes none 0 linear exp_results/ memory > exp_results/diabetes_none_linear_memory.txt
python experiments.py diabetes none 0 ae exp_results/ memory > exp_results/diabetes_none_ae_memory.txt
python experiments.py diabetes none 0 tsne exp_results/ memory > exp_results/diabetes_none_tsne_memory.txt
python experiments.py diabetes none 0 som exp_results/ memory > exp_results/diabetes_none_som_memory.txt


python experiments.py breastcancer shift 3 linear exp_results/ memory > exp_results/breastcancer_shift_linear_memory.txt
python experiments.py breastcancer shift 3 ae exp_results/ memory > exp_results/breastcancer_shift_ae_memory.txt
python experiments.py breastcancer shift 3 tsne exp_results/ memory > exp_results/breastcancer_shift_tsne_memory.txt
python experiments.py breastcancer shift 3 som exp_results/ memory > exp_results/breastcancer_shift_som_memory.txt

python experiments.py diabetes shift 3 linear exp_results/ memory > exp_results/diabetes_shift_linear_memory.txt
python experiments.py diabetes shift 3 ae exp_results/ memory > exp_results/diabetes_shift_ae_memory.txt
python experiments.py diabetes shift 3 tsne exp_results/ memory > exp_results/diabetes_shift_tsne_memory.txt
python experiments.py diabetes shift 3 som exp_results/ memory > exp_results/diabetes_shift_som_memory.txt

python experiments.py diabetes gaussian 3 linear exp_results/ memory > exp_results/diabetes_gaussian_linear_memory.txt
python experiments.py diabetes gaussian 3 ae exp_results/ memory > exp_results/diabetes_gaussian_ae_memory.txt
python experiments.py diabetes gaussian 3 tsne exp_results/ memory > exp_results/diabetes_gaussian_tsne_memory.txt
python experiments.py diabetes gaussian 3 som exp_results/ memory > exp_results/diabetes_gaussian_som_memory.txt

python experiments.py breastcancer gaussian 3 linear exp_results/ memory > exp_results/breastcancer_gaussian_linear_memory.txt
python experiments.py breastcancer gaussian 3 ae exp_results/ memory > exp_results/breastcancer_gaussian_ae_memory.txt
python experiments.py breastcancer gaussian 3 tsne exp_results/ memory > exp_results/breastcancer_gaussian_tsne_memory.txt
python experiments.py breastcancer gaussian 3 som exp_results/ memory > exp_results/breastcancer_gaussian_som_memory.txt

python experiments.py breastcancer zero 3 linear exp_results/ memory > exp_results/breastcancer_zero_linear_memory.txt
python experiments.py breastcancer zero 3 ae exp_results/ memory > exp_results/breastcancer_zero_ae_memory.txt
python experiments.py breastcancer zero 3 tsne exp_results/ memory > exp_results/breastcancer_zero_tsne_memory.txt
python experiments.py breastcancer zero 3 som exp_results/ memory > exp_results/breastcancer_zero_som_memory.txt

python experiments.py diabetes zero 3 linear exp_results/ memory > exp_results/diabetes_zero_linear_memory.txt
python experiments.py diabetes zero 3 ae exp_results/ memory > exp_results/diabetes_zero_ae_memory.txt
python experiments.py diabetes zero 3 tsne exp_results/ memory > exp_results/diabetes_zero_tsne_memory.txt
python experiments.py diabetes zero 3 som exp_results/ memory > exp_results/diabetes_zero_som_memory.txt
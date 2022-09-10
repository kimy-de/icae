i=2
echo ${i}
python3 train.py --latent_size ${i} --num_epochs 4000  --lr 1e-4

for i in 3 5 8 12
do
    echo ${i}
    python3 train.py --latent_size ${i} --num_epochs 3000
done
echo 'now centroids'
python3 centroids.py

for i in 2 3 5 8 12
do
    echo ${i}
    python3 icae_train.py --latent_size ${i} --num_clusters 5
    echo 'trajectory'
    python3 trajectory.py --latent_size ${i}
done

python3 dist2d3d.py
python3 evaluation.py

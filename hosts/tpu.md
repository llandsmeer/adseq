```
alias tpu='gcloud compute tpus tpu-vm'
name=my-tpu
zone=us-central2-b
project=ml-spike-event-queues

# 'v3-8'
name=my-tpu3
zone=europe-west4-a

# 'v2-8'
name=my-tpu2
zone=us-central1-f

tpu create $name \
    --zone=$zone \
    --project=$project  \
    --accelerator-type='v4-8' \
    --preemptible \
    --version=tpu-ubuntu2204-base

tpu scp \
    --compress \
    --recurse \
    --zone $zone \
    --project $project \
    benchmarks implementations *.py tests \
    $name:

tpu ssh \
    --zone "$zone" \
    --project $project \
    $name


pip install -U "jax[tpu]" 
pip install -U tqdm matplotlib pandas
pip install typing-extensions

screen

python3 benchmarks/profile_poisson.py; python3 benchmarks/profile_recurrent_snn.py

tpu scp \
    --compress \
    --zone $zone \
    --project $project \
    $name:ml_spike_event_queues/benchmarks/'*tpu_jax*.json' \
    out

tpu delete $name
```

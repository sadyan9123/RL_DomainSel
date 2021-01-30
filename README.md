### Instructions

- Assume that you have already install miniconda (https://docs.conda.io/en/latest/miniconda.html) and now create a python environment:

```bash
conda create --name py36rl python=3.6
source activate py36rl
```

- Some packages are necessary:

```bash
conda install -c conda-forge tensorflow=1.3 scikit-learn=0.20.2 pandas=0.23.4 matplotlib
```

- Download the benchmarks:

```bash
cd code/
wget https://github.com/sosy-lab/sv-benchmarks/archive/svcomp16.zip
unzip svcomp16.zip
# Now, we assume the unzipped directory is "svcomp16"
```

- Add the test set files to benchmark's directory:

```bash
cp sets/test_linux.set sets/test_spl.set code/svcomp16/c/
cp sets/challenge.set code/svcomp16/c/
```

- Java8 is required:

```bash
sudo apt-get install openjdk-8-jdk
```

- We use `benchexec` (version 1.16 is recommended) for benchmarking. Type the following script to install `benchexec`:

```bash
pip install benchexec==1.16
```

- Resolve the permission issues (required by `benchexec`):

```bash
sudo chmod o+wt '/sys/fs/cgroup/cpuset/' '/sys/fs/cgroup/memory/' '/sys/fs/cgroup/freezer/' '/sys/fs/cgroup/cpu,cpuacct/'
```

- Disable the swap partition for precise benchmarking  (required by `benchexec`):

```bash
sudo swapoff -a
```
<!-- 
- Start the experiment on the ordinary set (our approach, RL-Domainsel):
```bash
# Run the RL as the server
cd code/Q_learning
python TDD*.py --NN_dir NNPara20190719-174345_ForwardNet3LConfigable_a-0.1_s5_e0_b-2_p-15_True_0.9_10000000_2048_h512_aSigmoid --rich_feature True --port 5002
# Run the analyzer as the client
cd code/RL_Domainsel
cp RLServerInfo-rl.config RLServerInfo.config
benchexec rl_domainsel.xml
``` -->



- Start the experiment on the ordinary set (baseline, CPA-RefSel):

```bash
cd code/RL_Domainsel
cp RLServerInfo-baseline.config RLServerInfo.config
benchexec cpa-refsel.xml
```

- Start the experiment on the challenge set (our approach, RL-Domainsel):

```bash
# Run the RL as the server
cd code/Q_learning
python TDD*.py --NN_dir NNPara20190719-174345_ForwardNet3LConfigable_a-0.1_s5_e0_b-2_p-15_True_0.9_10000000_2048_h512_aSigmoid --rich_feature True --port 5002
# Run the analyzer as the client
cd code/RL_Domainsel
cp RLServerInfo-rl.config RLServerInfo.config
benchexec rl_domainsel_challenge.xml
```

*[^_^]:Only the name of the model is different in the command lines

*[^_^]:2 domain
- Start the experiment on the ordinary set (our approach, RL-Domainsel):
```bash
# Run the RL as the server
cd code/RL_2_domain
python TDD*.py --NN_dir NNPara20210124-093144_ForwardNet3LConfigable_a0.5_s5_e0_b-2_p-15_True_0.9_10000000_2048_h256_aSigmoid --rich_feature True --port 5002
# Run the analyzer as the client
cd code/RL_Domainsel
cp RLServerInfo-rl.config RLServerInfo.config
benchexec rl_domainsel.xml
```

*[^_^]:3 domain
- Start the experiment on the ordinary set (our approach, RL-Domainsel):
```bash
# Run the RL as the server
cd code/RL_3_domain
python TDD*.py --NN_dir NNPara20210127-171000_ForwardNet3LConfigable_a0.8_s5_e0_b-2_p-5_True_0.9_10000000_2048_h256_aSigmoid --rich_feature True --port 5002
# Run the analyzer as the client
cd code/RL_Domainsel
cp RLServerInfo-rl.config RLServerInfo.config
benchexec rl_domainsel.xml
```



```bash
conda create -n verl_custom python=3.11
```

Enter and install

```bash
conda activate verl_custom
cd verl_proj
pip install -e . -i https://mirrors.aliyun.com/pypi/simple
```

Install vllm

```bash
pip install vllm -i https://mirrors.aliyun.com/pypi/simple
```
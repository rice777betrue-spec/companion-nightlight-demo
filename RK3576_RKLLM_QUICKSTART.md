# RK3576 RKLLM 0.5B 文本接口

## 已准备的文件

- 板端服务目录：`.cache/deploy/rkllm-server-rk3576`
- 模型：`.cache/models/rk3576/Qwen2-0.5B-Instruct_RK3576_w4a16_g128.rkllm`
- 模型大小：`691738964` 字节
- SHA-256：`a166cdcff5d8a2e33423d01d8bac6f7a6ad1e1130fbd0bfefdb8b69082113e87`
- RKLLM Runtime / Toolkit：`1.3.0`

模型由社区作者 HanzoHuang 用 RKLLM Toolkit 1.3.0 转换，上游
Qwen2-0.5B-Instruct 标注 Apache-2.0。它适合开发板原型验证，
不是 Rockchip 官方模型库发布物。

## 1. 先检查开发板

```bash
cat /proc/device-tree/model
cat /etc/os-release
uname -a
free -h
ls -l /dev/rknpu /dev/dma_heap
dmesg | grep -Ei 'rknpu|rkllm' | tail -30
ip -4 addr
```

## 2. 从 Windows 传到板端

把 `<BOARD_USER>` 和 `<BOARD_IP>` 替换为真实值：

```powershell
scp -r "C:\Users\DELL\Documents\ChatGPT\公司\companion-nightlight-demo\.cache\deploy\rkllm-server-rk3576" <BOARD_USER>@<BOARD_IP>:/userdata/
scp "C:\Users\DELL\Documents\ChatGPT\公司\companion-nightlight-demo\.cache\models\rk3576\Qwen2-0.5B-Instruct_RK3576_w4a16_g128.rkllm" <BOARD_USER>@<BOARD_IP>:/userdata/
```

## 3. 在板端启动服务

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv /userdata/rkllm-venv
source /userdata/rkllm-venv/bin/activate
pip install Flask==2.2.2 Werkzeug==2.2.2

cd /userdata/rkllm-server-rk3576
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 flask_server.py \
  --rkllm_model_path /userdata/Qwen2-0.5B-Instruct_RK3576_w4a16_g128.rkllm \
  --target_platform rk3576
```

服务会监听 `0.0.0.0:8080`。它没有真实身份验证，首版只应暴露在
可信局域网。

## 4. 验证文本接口

在板端或其他局域网设备上：

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rkllm","messages":[{"role":"user","content":"你好，请用一句中文介绍你自己"}],"stream":false,"top_k":1,"max_tokens":64}'
```

在当前 Windows 项目上：

```powershell
.\.venv\Scripts\python.exe .\rkllm_text_smoke.py --server http://<BOARD_IP>:8080
```

## 常见问题

- `libomp.so: cannot open shared object file`：先运行
  `ldd lib/librkllmrt.so | grep 'not found'`，再从板卡 SDK/交叉编译工具链中
  取与系统架构匹配的 `libomp.so`。
- 没有 `/dev/rknpu`：先修复内核 NPU 驱动，HTTP 服务代码无法规避。
- 电脑无法访问 `8080`：检查板端 IP、防火墙和两台设备是否在同一局域网。

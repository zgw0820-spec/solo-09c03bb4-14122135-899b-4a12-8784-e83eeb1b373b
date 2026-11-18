# 风机停机预警训练平台（可续跑）

用 FastAPI + NumPy + SQLite 实现的风机停机预警逻辑回归训练平台。训练由两个工作器
容器执行，每个批次都把权重、动量、损失和进度作为**一个原子断点**提交；工作器退出后
另一个工作器自动接管，并从最近断点继续。最终权重与不中断训练逐位一致。

## 组成

| 组件 | 说明 |
| --- | --- |
| `api` | FastAPI，提交任务、查询进度/断点/权重、Swagger 演示 |
| `worker-1` / `worker-2` | 同一镜像的 `python -m app.worker`，共享同一个 SQLite 文件 |
| `training-data` | Docker 卷，挂载到三个容器的 `/data` |

两个工作器共享**同一个** SQLite 数据库（`/data/training.db`），因此"检查租约并写入
断点"和"写入断点"是同一个 `BEGIN IMMEDIATE` 事务，不需要跨库协调。

## 训练约定

- 输入 8～64 条记录，每条是风机振动/电流等特征，标签 `0`=正常、`1`=停机预警；
  各特征列按提交样本标准化后追加偏置列，权重长度为 `特征数 + 1`。
- 轮数 2～5；按输入顺序每 8 条一批（最后一批可不足 8 条）。
- 逻辑回归（sigmoid + 二元交叉熵），SGD，学习率 `0.1`，动量 `0.9`，float64。
- 每批提交的断点包含：`step`、`training_epoch`、`batch_in_epoch`、`weights`、
  `momentum_velocity`、`batch_loss`、`loss`（全量交叉熵）、`progress`、提交时间、
  `worker_id`、`lease_epoch`。标签更正会重算保留断点的 `loss`，并删除之后由旧标签
  算出的断点。

## 启动

```bash
docker compose up -d --build
# Swagger: http://localhost:8000/docs
```

`worker-1` / `worker-2` 故意不设 `restart` 策略，便于演示接管。被 kill 的工作器用
`docker compose start worker-1` 恢复（它会以空闲状态重新加入，不会抢占已完成任务）。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/jobs` | 提交 `{"features": [[...]], "labels": [...], "epochs": 3}`，返回 201 与任务 |
| `GET` | `/jobs` | 最近任务列表 |
| `GET` | `/jobs/{job_id}` | 已确认步数、进度、损失、标签、当前权重；完成时含 `final_weights` |
| `GET` | `/jobs/{job_id}/checkpoints` | 全部批次断点（可按 `after_step` 增量拉取） |
| `POST` | `/jobs/{job_id}/corrections` | 更正一个样本标签，见下文“标签更正” |
| `GET` | `/jobs/{job_id}/corrections` | 该任务的更正历史 |
| `GET` | `/jobs/{job_id}/events` | 审计事件：创建、开始/从第几步续训、标签更正、完成、丢失租约 |
| `GET` | `/workers` | 工作器心跳与租约持有者 |
| `GET` | `/health` | 健康检查 |

`Swagger`（`/docs`）里直接 `POST /jobs` 提交即可看到训练开始；`GET /jobs/{id}` 与
`GET /jobs/{id}/checkpoints` 查看进度与每个断点的权重/动量/损失。

## 标签更正

风机停机样本可能标错。任务仍为 `queued`/`running` 时，可更正任意一个样本：

```bash
curl -X POST http://localhost:8000/jobs/<job_id>/corrections \
  -H 'Content-Type: application/json' \
  -d '{"record_index": 9, "old_label": 0, "new_label": 1}'
```

Swagger（`/docs`）中对应 `POST /jobs/{job_id}/corrections`。`record_index` 是提交
顺序中的零基下标，`old_label` 是当前标签（乐观校验，不匹配返回 `409`），任务已完成
返回 `409`，下标或标签越界返回 `422`。

更正按“回滚该样本对训练的影响”执行，而不是只改一行标签：

1. 固定分批下，样本 `i` 首次进入批次 `i // 8 + 1`（每个 epoch 顺序相同）。
2. 若该批次已提交，则保留它**之前**的断点（`step - 1`）的权重与动量并把进度退回到
   该步，删除之后所有断点；此前的权重、动量沿用，不重算。
3. 若该批次尚未执行，进度不动，只改标签。
4. 所有保留断点的 `loss`（全量交叉熵）都用新标签重算，与该断点的权重/动量及
   `jobs.loss` 保持一致；`batch_loss`、权重与动量仍对应原批次。
5. 改标签、删断点、重算损失、写更正历史与审计事件在同一个 `BEGIN IMMEDIATE`
   事务内完成。

因此从保留断点继续训练后，最终权重与“用更正后的样本从头训练”逐位一致；被废止断点
的旧标签结果不会再被续跑使用。

**与批次提交 / 工作器接管的交错。** 更正与批次提交都是同一数据库的写事务，SQLite
串行化它们：批次要么先提交（随后被更正的保留断点判定回滚），要么等更正提交后再执行
（在自己的事务里重新读取新标签）。被接管而失去租约的旧工作器仍受 `(worker_id,
lease_epoch)` 围栏约束，无法在更正后写入旧标签算出的结果。故障（进程被杀、容器重启）
后，新工作器从保留断点续跑，标签更正已持久化，不丢失。

## 续跑与防回退机制

1. 工作器每批在**同一个事务**里写检查点、更新任务进度、刷新租约心跳。任何一项失败
   则整体回滚，不会出现"权重已变但进度没变"的中间态。
2. 租约保存在同一数据库的 `leases` 表。工作器每 `1.5s` 心跳；持有者心跳超过
   `6s`（`TRAINING_LEASE_SECONDS`）未更新，另一个工作器才可接管，接管时
   `epoch + 1`。
3. 每次提交前在事务内校验 `(worker_id, epoch)` 仍是租约持有者。被接管后仍存活（或
   迟到唤醒）的旧工作器校验失败，直接 `LeaseLost` 退出：既不能回退断点，也不能把
   别人的完成状态改写，更不能重复提交完成事件。
4. 完成状态由 `step == total_steps` 在提交事务内一次性置位；旧工作器无法再写入。
5. SQLite 使用 WAL + `synchronous=FULL`，已提交的断点在进程崩溃/容器重启后仍然存在。

## 演示与验证

```bash
# 完整演示（Docker）：提交 -> 观察 -> kill 当前持有者 -> 接管续跑 -> 对比不中断训练
python3 scripts/demo.py

# 标签更正演示：训练中更正一个样本 -> 自动回滚/续跑 -> 与“更正后从头训练”比对
python3 scripts/demo_correction.py

# 没有 Docker 时用本机进程演示同样流程
python3 scripts/demo.py --local
python3 scripts/demo_correction.py --local

# 不依赖 Docker 的自检（含更正 + 接管）
python3 scripts/selftest.py

# 单元/集成测试（含真实 uvicorn + 两个工作器进程、SIGKILL 接管、迟到工作器围栏、
# 训练中标签更正与故障接续）
python3 -m pytest -q
```

`demo.py` 第 6 步会用同一份输入再训练一次（不中断），并断言两次的 `final_weights`、
`loss`、`momentum_velocity` 完全相等。`demo_correction.py` 则在训练中途更正一个样本，
等待回滚后自动续跑完成，再与“用更正后数据从头训练”逐位比对。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TRAINING_DATA_DIR` | `./data` | 数据库目录 |
| `TRAINING_DB_PATH` | `$TRAINING_DATA_DIR/training.db` | 共享数据库 |
| `TRAINING_LEASE_SECONDS` | `6` | 租约超时 |
| `TRAINING_HEARTBEAT_INTERVAL` | `1.0` | 心跳间隔（必须小于租约） |
| `TRAINING_POLL_INTERVAL` | `0.3` | 空闲轮询/批间间隔 |
| `TRAINING_BATCH_DELAY_SECONDS` | `0` | 模拟每批耗时，便于演示 |
| `TRAINING_DEMO_PAUSE_FILE` | 未设置 | 演示用：文件存在时在批间暂停（不心跳） |
| `TRAINING_DEMO_CRASH_AFTER` | 未设置 | 演示用：指定步数提交后 `os._exit` |

## 目录

```
app/main.py       API 与 Swagger
app/worker.py     工作器主循环、租约续训
app/store.py      任务状态机、批次提交、断点查询、标签更正
app/lease.py      租约获取/续约/围栏校验
app/model.py      逻辑回归、交叉熵、SGD + 动量
app/features.py   输入校验与标准化
app/db.py         SQLite 连接与表结构
tests/            单元、API、进程级集成测试
scripts/demo.py   Swagger 同款接口的端到端演示（故障接管）
scripts/demo_correction.py  标签更正端到端演示
scripts/selftest.py  无 Docker 自检
```

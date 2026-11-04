"""Event-Graph 长距离记忆（v1.3 最终实现规范）。

设计文档：Papers/pi05_GNN_LongTerm_Memory_Final_Implementation_v1.3.docx
（配套 v1.0 总体方案 / v1.1 工程决策 / v1.2 工程补充）

平面划分（v1.3 §1）：

    CONTROL PLANE  —— 完全无参数、stop-gradient、deterministic
        只消费 frozen PCA frame summary + state + timestamp + phase；
        产出 event metadata（boundary / L=8 frame ids / rep id / node mask /
        boundary_strength / provenance），不做任何可学习决策。
        唯一入口 EventGraphController，四个调用方（预处理、训练回放、在线
        mem_buffer/policy server、诊断脚本）必须共用。

        ⚠️ 边界判定已从 v1.3 §2.2/§2.3 的「三项证据 + 滑动门槛」换成「安静点切分」：
        真值 is_subgoal_boundary 实测标在机械臂**最安静**的那一帧上，而原方案的
        三项证据全是「变化越大分越高」，方向正好相反（实测命中率 9.1%，比随机还差）。
        详见 controller.py 模块头。原标定脚本 eventgraph_calibrate.py 已废弃。

    DATA PLANE     —— 可学习，接 action loss
        L=8 的 512-d FP16 frozen tokens + state
          → v_pool → frame fusion → Event GRU → h_event
          → 2 层 relation-aware GNN
          → 24 latent slots soft read 全部有效长期节点
          → 24×16 = 384 long-term tokens
          + 8 recent frames × 16 = 128 tokens
          → 512 memory tokens → π0.5 Action Expert

模块划分对应 v1.3 §10 的执行阶段：

    config.py          GraphMemoryConfig    —— 唯一配置源（§8）
    frame_summary.py   FrozenFrameSummary   —— Phase 1：冻结 PCA + f_t（§2.1）
    store.py           EventFeatureStore    —— Phase 1：episode 级连续存储
    controller.py      EventGraphController —— Phase 2/3：安静点切分 / medoid /
                                              pending / 晋升 / 相邻节点合并
    memory.py                                —— Phase 4/5：Data Plane 的事件编码器
                                              与 GNN（接 action loss）
    readout.py         pack_eg_tensors      —— 把「这一帧看见了哪些 Event」拼成
                                              模型要的 15 个 eg_* 张量。离线 dataset
                                              与在线 buffer **共用这一个函数**
    online.py          EventGraphMemoryBuffer —— Phase 6：policy server 侧逐帧增量
                                              跑同一个 Controller（不许回退
                                              frame_sampling，§11）

Phase 1~6 都已实现。§11 的门 —— 离线落盘与在线逐帧逐位一致 —— 由
`scripts/diag/eventgraph_p6_parity.py` 把关；那道测试没过之前，不许报告 graphmem
的 RoboMME eval 数字。
"""

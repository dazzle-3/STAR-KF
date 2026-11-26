# 当前 InitialPrior 诊断与统计语义

每轮 `result.iterations{q}` 保留现有正式产品族。下表覆盖本次改变的字段。

| 字段 | 单位/尺寸 | 当前语义 |
|---|---|---|
| `Y_q` | mm，点×历元 | `G(phi-tau_prev)` |
| `observation_diagnostics.operators.G` | mm/rad，历元×IFG×有效观测模式 | 式（2b）恢复算子，含数值加载/弱岭影响 |
| `...SigmaPhi` / `...SigmaY` | rad² / mm² | 未加载物理相位协方差及式（3b）传播；模式由 `pattern_index` 指定 |
| `D1_q`, `P1_q` | mm / mm² | KF1 平滑组合与 `diag(A1 SigmaY A1')` |
| `E1_q` | mm | `L1 Y` |
| `D2_raw_q`, `P2_q` | mm / mm² | KF2 完整残差轨迹候选与 `diag(F2 SigmaY F2')` |
| `WD`, `WU`, `WA`, `aM`, `W2` | 点×1，无量纲 | 式（11）—（14）；旧五分量不再存在于正式输出 |
| `D2_q`, `D_joint_q` | mm | `W2.*D2_raw`；`D1+D2` |
| `P_joint_q` | mm² | `diag((A1+wF2) SigmaY (A1+wF2)')`，保留共享观测交叉项 |
| `R_atm_q` | rad，点×IFG | 原始 IFG 减去本轮联合形变 |
| `tau_candidate_q`, `tau_candidate_var_q` | rad / rad² | 同一预测系数的空间均值和潜在场误差；后者为线性凸融合工作界 |
| `a_q`, `tau_q` | rad，点×历元 / 点×IFG | 参考相对历元场，及严格 `a_q*B'` |
| `R_current`, `R_next` | mm²，1×历元 | 本轮和下一轮原始历元工作方差，参考分量不强制为零 |
| `R_next_components` | mm² / rad² | 四项反馈、原始/有界目标、独立映射 `Gsp/Gres`、求解下限与正则 |
| `Delta_D`, `Delta_tau`, `Delta_R` | 无量纲 | 固定共同支持上的 Frobenius 相对变化 |
| `convergence.C`, `C_diagnostic` | 无量纲 | 前三项最大值；兼容名称 `C_diagnostic` 现在也表示正式判据 |
| `Delta_E` | 无量纲 | 残差能量变化诊断，不进入停止最大值 |
| `convergence.support_lost`, `streak`, `terminate` | 布尔/整数 | 支持丢失不能计稳定；第3轮起计入连续合格轮 |
| `operators`, `operator_pixel_indices` | 固定审计像元 | 全部拟合算子及条件矩；大规模默认首像元，小规模保留全部 |

审计算子中的 `KF1{m}.state_var_mm2` 和 `KF2.models{m}.posterior_diagnostic` 是高斯参考条件/状态协方差诊断，不是 `P1/P2/P_joint`。`KF2` 保存 `L1,RE,WE,A2,F2,C2`、三个 `Kd/Kdelta/A2`、保留行 `Wm`、`Gamma`、创新、`S0`、膨胀和门控；拒绝行保留评分。KF1 的 `A` 同时记录参考化和 RTS 的固定线性影响。

VCE 保存 `unfloored_components` 和每个分量初始一次加十次重加权的 `solver_history`，包括原目标、非负 KKT、权重、尺度和解。`sigma2_slc` 的 `.03 rad²` 是优化之后的既有下限。原始历元分量、双轴参考协方差、完整 IFG 协方差分别保存；正则唯一性不等于数据可辨识性或整体重加权最优性。

`kriging.support_mask` 标记空间支持；无支持返回 NaN，正式主循环不以补洞伪造证据。预测噪声 `c0_ifg+vI` 只计一次，加载仅影响系数；潜在误差不再扣 nugget。`loading_rad2` 记录每窗口/IFG 实际加载。

`R_next_components` 的四项共享拟合数据，可能重叠，省略跨分量、跨 IFG、跨迭代协方差；网络映射还是对角 IFG 近似，不保证保留空间局部上界。`mapper_phase` 在相位域，rad²→mm² 仅在外层转换一次。

最终 `criterion_fulfilled` 和 `iteration_limit_reached` 分开记录；`termination_reason` 给实际退出原因。`finalization.d_final` 仅比较最终条件形变和末轮形变，不改变既有停止标志。`finalization.spatial_update_performed=false`，其协方差用最终条件估计自身的算子重算。

收敛诊断中，`criterion_fulfilled`仍表示最后完成轮次的判据状态；`criterion_ever_fulfilled`表示观察期间是否曾满足；`criterion_first_satisfied_iteration`记录首次应停轮次，0表示从未满足。`diagnostic_continue_after_convergence`标记是否仅为诊断而继续观察；默认关闭。开启时实际退出原因是`diagnostic_iteration_limit`，与正常模式的`joint_state_criterion`或`iteration_limit`区分。

`evaluation/evaluate_iteration_history.m`仅在估计完成后使用真值生成逐轮CSV/JSON，分别标记`outer_end`和`final_if_stop`。这些指标不进入主循环、模型选择或停止逻辑；主要精度统一排除确定性参考历元并对大气真值参考化。


本版本的默认阈值为`cfg.convergence.joint_state_tol=0.01`，从第3轮起计数、要求连续2轮满足判据、最多5轮；默认不诊断续跑。当前仅保留`joint_state`策略，已弃用的条件分块、混合容差、差异原则、准最优和L曲线停止分支均已清理。`Delta_D`对应外循环本轮时序形变，`C=max(Delta_D,Delta_tau,Delta_R)`。`runtime_profile.conditional_stopping_pass_sec`仅为兼容保留的零值字段，正式路径不预先执行下一轮时序估计。最终条件估计仍使用自身拟合算子与协方差。

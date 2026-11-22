| Condition | Metric | Comparison | n | Mean (unc.-aware) | Mean (other) | Change (%) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| steady | intrusion_rate | predictive_unc vs snapshot | 270 | 0.180 | 0.242 | -25.7 | 5.8e-21 | 3.5e-20 |
| steady | intrusion_rate | predictive_unc vs predictive | 270 | 0.180 | 0.202 | -11.1 | 2.2e-08 | 1.1e-07 |
| erratic | intrusion_rate | predictive_unc vs snapshot | 90 | 0.261 | 0.302 | -13.5 | 9.5e-04 | 0.004 |
| erratic | intrusion_rate | predictive_unc vs predictive | 90 | 0.261 | 0.286 | -8.7 | 0.007 | 0.013 |
| reactive | intrusion_rate | predictive_unc vs snapshot | 90 | 0.273 | 0.313 | -12.6 | 0.002 | 0.006 |
| reactive | intrusion_rate | predictive_unc vs predictive | 90 | 0.273 | 0.296 | -7.6 | 0.023 | 0.023 |
| steady | mean_min_clearance | predictive_unc vs snapshot | 270 | 1.030 | 0.971 | +6.1 | 3.6e-04 | 0.002 |
| steady | mean_min_clearance | predictive_unc vs predictive | 270 | 1.030 | 1.003 | +2.7 | 0.030 | 0.148 |
| erratic | mean_min_clearance | predictive_unc vs snapshot | 90 | 0.808 | 0.822 | -1.7 | 0.524 | 0.551 |
| erratic | mean_min_clearance | predictive_unc vs predictive | 90 | 0.808 | 0.826 | -2.1 | 0.212 | 0.551 |
| reactive | mean_min_clearance | predictive_unc vs snapshot | 90 | 0.776 | 0.818 | -5.1 | 0.037 | 0.148 |
| reactive | mean_min_clearance | predictive_unc vs predictive | 90 | 0.776 | 0.805 | -3.5 | 0.184 | 0.551 |
| steady | mean_time_to_goal | predictive_unc vs snapshot | 270 | 11.098 | 9.172 | +21.0 | 4.7e-36 | — |
| steady | mean_time_to_goal | predictive_unc vs predictive | 270 | 11.098 | 9.744 | +13.9 | 2.7e-28 | — |
| erratic | mean_time_to_goal | predictive_unc vs snapshot | 90 | 8.285 | 7.083 | +17.0 | 1.2e-10 | — |
| erratic | mean_time_to_goal | predictive_unc vs predictive | 90 | 8.285 | 7.515 | +10.3 | 3.1e-06 | — |
| reactive | mean_time_to_goal | predictive_unc vs snapshot | 90 | 8.264 | 7.122 | +16.0 | 1.1e-08 | — |
| reactive | mean_time_to_goal | predictive_unc vs predictive | 90 | 8.264 | 7.581 | +9.0 | 9.0e-06 | — |

Pre-registered verdict: {'H5_supported_by_condition': {'steady': True, 'erratic': True, 'reactive': True}, 'H6_supported_by_condition': {'steady': True, 'erratic': True, 'reactive': True}, 'robustness_claim_allowed': True}

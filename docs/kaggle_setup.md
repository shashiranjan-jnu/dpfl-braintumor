# Kaggle Setup

Check date  2026-09-30 (YYYY-MM-DD)

## Account limits
| Item | Value |
| --- | --- |
| Weekly GPU quota (hours) | 30 |
| Maximum session length (hours) | 12 |
| T4 x2 billing (quota per wall-clock hour) | 1 |
| Phone verification | Done |

## Secrets
| Label | Purpose | Scope | Expires |
| --- | --- | --- | --- |
| kaggle_dpfl-braintumor | GitHub push token | dpfl-braintumor only, Contents read and write | 2027-09-30|

## Launcher rules
- Internet On, notebook Private, secret attached
- Code copy at a tag in /tmp/code, results copy of main in /tmp/results, token only in the GIT_PUSH_URL environment variable
- Accelerator None for CPU-only work

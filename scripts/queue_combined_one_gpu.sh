#!/bin/bash
# COMBINED MeLoMIA-TabSyn restricted to GPU 0 (Steven, 2026-10-10): let the three
# GPU-0 shadow loops finish K=10, then run the remaining stages on GPU 0 only.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
for p in 4137535 4137536 4137538; do while kill -0 $p 2>/dev/null; do sleep 120; done; done
./scripts/run_melomia_tabsyn.sh COMBINED 3 0 > logs/melomia_tabsyn/run_combined_gpu0.log 2>&1
echo "### [$(date -u +%m-%d\ %H:%M)] QUEUE_DONE (COMBINED, GPU 0 only)" >> logs/melomia_tabsyn/queue.log

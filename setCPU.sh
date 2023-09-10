#!/bin/bash
for i in {0..23}
do
  sudo cpufreq-set -c $i -g performance
done
echo "Govenors set to performance"

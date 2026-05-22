#!/usr/bin/env bash
# Try multiple parses of the same string against Replicate.
# Set REPLICATE_TOKEN in your environment before running.
: "${REPLICATE_TOKEN:?Set REPLICATE_TOKEN env var}"

for tok in \
  "${REPLICATE_TOKEN}" \
  "${REPLICATE_TOKEN}t" \
  "${REPLICATE_TOKEN}try" \
  "${REPLICATE_TOKEN%?}"
do
  status=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Token ${tok}" \
    https://api.replicate.com/v1/account)
  echo "len=${#tok} status=${status}"
done

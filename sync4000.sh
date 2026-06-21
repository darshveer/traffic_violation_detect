#!/bin/bash
# Run commands / copy files to the RTX 4000 Ada box (ubuntu@10.0.1.17) via the
# oem box (10.10.0.206) as a jump host. Both hops use the same password.
set -u
PW='root@123'
JUMP='oem@10.10.0.206'
TARGET='ubuntu@10.0.1.17'
SO='-o StrictHostKeyChecking=no -o ConnectTimeout=20'
filt(){ grep -aviE "post-quantum|store now|upgraded|openssh|Warning: Perm"; }

case "${1:-}" in
  run) shift
    enc=$(printf '%s' "$*" | base64)
    sshpass -p "$PW" ssh $SO "$JUMP" "sshpass -p '$PW' ssh $SO $TARGET \"echo $enc | base64 -d | bash\"" 2>&1 | filt ;;
  pushfile) shift  # local -> remote abs path on 4000
    local_f="$1"; remote_f="$2"
    base64 < "$local_f" | sshpass -p "$PW" ssh $SO "$JUMP" "sshpass -p '$PW' ssh $SO $TARGET \"base64 -d > '$remote_f'\"" 2>&1 | filt ;;
  getfile) shift  # remote abs path on 4000 -> local
    remote_f="$1"; local_f="$2"
    sshpass -p "$PW" ssh $SO "$JUMP" "sshpass -p '$PW' ssh $SO $TARGET \"base64 < '$remote_f'\"" 2>&1 | filt | base64 -d > "$local_f" ;;
  *) echo "usage: run|pushfile|getfile" ;;
esac

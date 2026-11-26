#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HADOOP_HOME:?HADOOP_HOME is not set}"
java_home="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}"
data="${IMGCNN_HDFS_DIR:-$HOME/imgcnn-hdfs}"
nm_mb="${IMGCNN_NM_MB:-3072}"
nm_vcores="${IMGCNN_NM_VCORES:-4}"
conf="$HADOOP_HOME/etc/hadoop"
bin="$HADOOP_HOME/bin"

render() {
  sed -e "s|@DATA_DIR@|$data|g" \
      -e "s|@HADOOP_HOME@|$HADOOP_HOME|g" \
      -e "s|@JAVA_HOME@|$java_home|g" \
      -e "s|@NM_MB@|$nm_mb|g" \
      -e "s|@NM_VCORES@|$nm_vcores|g" \
      -e "s|@USER@|$(id -un)|g" "$here/conf/$1" > "$conf/$1"
}

install() {
  mkdir -p "$data"
  for f in core-site.xml hdfs-site.xml yarn-site.xml mapred-site.xml capacity-scheduler.xml hadoop-env.sh; do
    render "$f"
  done
  if [ ! -d "$data/name/current" ]; then
    "$bin/hdfs" namenode -format -nonInteractive -force
  fi
  rm -rf "$here/build"
  mkdir -p "$here/build/classes"
  "$java_home/bin/javac" --release 17 -cp "$("$bin/hadoop" classpath)" -d "$here/build/classes" "$here/SplitOutputFormat.java"
  "$java_home/bin/jar" cf "$here/build/imgcnn-hadoop.jar" -C "$here/build/classes" .
  echo "config in $conf, hdfs data in $data, jar in $here/build/imgcnn-hadoop.jar"
}

daemon() {
  if ! "$bin/$1" --daemon status "$2" 2>/dev/null | grep -q "is running"; then
    "$bin/$1" --daemon start "$2"
  fi
}

start_hdfs() {
  daemon hdfs namenode
  daemon hdfs datanode
  for _ in $(seq 60); do
    if "$bin/hdfs" dfsadmin -safemode get >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  "$bin/hdfs" dfsadmin -safemode wait
  "$bin/hdfs" dfs -mkdir -p "/user/$(id -un)"
}

start_yarn() {
  daemon yarn resourcemanager
  daemon yarn nodemanager
  daemon mapred historyserver
  for _ in $(seq 90); do
    if "$bin/yarn" node -list 2>/dev/null | grep -q RUNNING; then
      echo "hdfs and yarn are up"
      return 0
    fi
    sleep 2
  done
  echo "nodemanager did not register" >&2
  return 1
}

stop_yarn() {
  "$bin/mapred" --daemon stop historyserver || true
  "$bin/yarn" --daemon stop nodemanager || true
  "$bin/yarn" --daemon stop resourcemanager || true
}

stop_hdfs() {
  "$bin/hdfs" --daemon stop datanode || true
  "$bin/hdfs" --daemon stop namenode || true
}

case "${1:-}" in
  install) install ;;
  start) start_hdfs; start_yarn ;;
  stop) stop_yarn; stop_hdfs ;;
  start-hdfs) start_hdfs ;;
  start-yarn) start_yarn ;;
  stop-yarn) stop_yarn ;;
  status) "$java_home/bin/jps" | grep -v Jps || true ;;
  *) echo "usage: $0 install|start|stop|start-hdfs|start-yarn|stop-yarn|status" >&2; exit 2 ;;
esac

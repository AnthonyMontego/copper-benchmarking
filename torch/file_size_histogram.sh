#!/bin/bash -l
# Characterize the file-size distribution of a directory (e.g. the torch install).
# Buckets files by size range and reports count + total bytes per bucket.
# Per Kaushik: multiply these counts by total ppn to project aggregate load at scale.
# Usage: ./file_size_histogram.sh <dir>   (default: lus_custom_pip_env)

DIR="${1:-/flare/hpc-spectacle/lus_custom_pip_env}"

echo "file-size distribution under: ${DIR}"
echo ""

find "${DIR}" -type f -printf '%s\n' | awk '
function human(b){
  if(b>=1073741824) return sprintf("%.2f GB", b/1073741824)
  if(b>=1048576)    return sprintf("%.2f MB", b/1048576)
  if(b>=1024)       return sprintf("%.2f KB", b/1024)
  return sprintf("%d B", b)
}
{
  n++; total+=$1
  if($1 < 1024)               {b["1_<1KB"]++;      sz["1_<1KB"]+=$1}
  else if($1 < 16384)         {b["2_1-16KB"]++;    sz["2_1-16KB"]+=$1}
  else if($1 < 262144)        {b["3_16-256KB"]++;  sz["3_16-256KB"]+=$1}
  else if($1 < 1048576)       {b["4_256KB-1MB"]++; sz["4_256KB-1MB"]+=$1}
  else if($1 < 10485760)      {b["5_1-10MB"]++;    sz["5_1-10MB"]+=$1}
  else if($1 < 104857600)     {b["6_10-100MB"]++;  sz["6_10-100MB"]+=$1}
  else                        {b["7_>100MB"]++;    sz["7_>100MB"]+=$1}
}
END{
  printf "%-14s %10s %14s %8s\n", "size range", "files", "total bytes", "% files"
  printf "%-14s %10s %14s %8s\n", "----------", "-----", "-----------", "-------"
  n_labels=split("1_<1KB 2_1-16KB 3_16-256KB 4_256KB-1MB 5_1-10MB 6_10-100MB 7_>100MB", order, " ")
  for(i=1;i<=n_labels;i++){
    k=order[i]
    if(b[k]>0){
      label=k; sub(/^[0-9]_/,"",label)
      printf "%-14s %10d %14s %7.1f%%\n", label, b[k], human(sz[k]), 100*b[k]/n
    }
  }
  print ""
  printf "TOTAL: %d files, %s\n", n, human(total)
  printf "median file: run separately if needed\n"
}
'

echo ""
echo "median file size:"
find "${DIR}" -type f -printf '%s\n' | sort -n | awk '{a[NR]=$1} END{
  m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2
  if(m>=1048576) printf "  %.2f MB\n", m/1048576
  else if(m>=1024) printf "  %.2f KB\n", m/1024
  else printf "  %d B\n", m
}'

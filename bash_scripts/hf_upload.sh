
# 仓库类型：model / dataset / space
REPO_TYPE="${REPO_TYPE:-model}"

# Hugging Face 仓库 ID，比如：yourname/your-model-repo
REPO_ID="${REPO_ID:-yourname/your-model-repo}"

# 本地文件或文件夹路径（相对或绝对都可以）
LOCAL_PATH="${LOCAL_PATH:-./my_model.bin}"

# 仓库中希望存放的路径（相对 repo 根目录）
REMOTE_PATH="${REMOTE_PATH:-my_model.bin}"

# 可选：提交信息
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload from upload_hf.sh}"

#######################################################

echo "================= Hugging Face Upload ================="
echo " REPO_TYPE   = ${REPO_TYPE}"
echo " REPO_ID     = ${REPO_ID}"
echo " LOCAL_PATH  = ${LOCAL_PATH}"
echo " REMOTE_PATH = ${REMOTE_PATH}"
echo " COMMIT_MSG  = ${COMMIT_MESSAGE}"
echo "======================================================="

# 真正上传的命令
huggingface-cli upload \
  --repo-type "${REPO_TYPE}" \
  "${REPO_ID}" \
  "${LOCAL_PATH}" \
  "${REMOTE_PATH}" \
  --commit-message "${COMMIT_MESSAGE}"

echo "上传完成！"

URLS_PATH=$1

for url_file in $(cat $URLS_PATH); do
    IFS='|' read -r url file <<< "$url_file"
    if [[ -z "$file" ]]; then
        continue
    fi
    echo "Downloading $file from $url"
    python -m aria2p add "$url" -o out="$file" &
done

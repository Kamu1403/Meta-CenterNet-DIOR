dir='Horizontal_Bounding_Boxes'
file=`/bin/ls -1 "$dir" | sort --random-sort | head -1`
path=`readlink --canonicalize "$dir/$file"` # Converts to full path
echo "The randomly-selected file is: $path"
cat $path
echo -e "\033[31mresults:"
cat $path | grep -G 'baseballfield\|basketballcourt\|bridge\|chimney\|ship'
echo -e "\033[0m"
#!/bin/bash
git branch
echo ""
git status
echo ""
echo ""
echo ""
echo "Please enter commit log"
read log 
echo ""
echo $log 
echo ""
echo ""

echo "Sure to add and commit" 
read confirm

echo ""

git add --all
echo ""
echo ""
git commit -m "$log" 

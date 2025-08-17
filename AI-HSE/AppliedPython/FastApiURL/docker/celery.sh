#!/bin/bash

cd src
# планировщик
if [[ "${1}" == "celery" ]]; then
  celery --app=tasks.tasks:celery worker -B -l INFO
# UI мониторинга задач
elif [[ "${1}" == "flower" ]]; then
  celery --app=tasks.tasks:celery flower
elif [[ "${1}" == "beat" ]]; then
  celery --app=tasks.tasks:celery beat -l INFO
 fi
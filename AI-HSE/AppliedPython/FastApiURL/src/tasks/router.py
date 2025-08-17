from fastapi import APIRouter
from .tasks import send_email

router = APIRouter(prefix="/report")


@router.get("/send")
def send_email_handler():
    try:
        # background_tasks.add_task(send_email, 'Sergei')
        send_email.apply_async(args=['Sergei'])
    except Exception as e:
        return {
            "status": 503,
            "details": str(e),
        }

    return {
        "status": 200,
        "details": "All ok"
    }



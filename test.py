import asyncio
from app.core.config import get_settings
from app.core.schemas import ChatRequest, RouterOutput
from app.core.constants import Intent
from app.session.session_store import SessionStore
from app.queueing.request_queue import QueueManager

async def main():
    settings = get_settings()
    print('settings:', settings.app_name, settings.llm_backend)

    req = ChatRequest(text='Cho anh một ly bạc xỉu đá')
    print('chat request:', req.model_dump())

    out = RouterOutput(action=Intent.ORDER, confidence=0.99)
    print('router json:', out.to_required_json())

    store = SessionStore()
    sess = await store.get_or_create()
    await store.add_user_message(sess.session_id, 'Có món nào ngon không?')
    await store.add_assistant_message(sess.session_id, 'Dạ, em có thể gợi ý vài món ạ.')
    loaded = await store.get(sess.session_id)
    print('session:', loaded.session_id, len(loaded.history))

    qm = QueueManager()

    async def fake_job(x):
        await asyncio.sleep(0.05)
        return x * 2

    result = await qm.router.run(fake_job, 21)
    print('queue result:', result)
    print('queue stats:', qm.stats())

asyncio.run(main())
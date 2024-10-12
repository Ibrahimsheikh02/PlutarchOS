import aioredis
import ssl
import os

async def get_redis_connection():
    redis_url = os.environ.get('REDIS_URL')

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    return await aioredis.create_redis_pool(
        redis_url,
        ssl=ssl_context
    )
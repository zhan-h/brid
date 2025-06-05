import logging
from django.utils.deprecation import MiddlewareMixin


# 获取自定义的日志记录器
logger = logging.getLogger('deepseek_user')


class LogIPAddressMiddleware(MiddlewareMixin):
    def process_request(self, request):
        logger.debug("Entering process_request")
        ip_address = request.META.get('REMOTE_ADDR')
        if ip_address:
            logger.debug(f"Request from IP: {ip_address}")
        else:
            logger.debug("REMOTE_ADDR is not found in request.META")


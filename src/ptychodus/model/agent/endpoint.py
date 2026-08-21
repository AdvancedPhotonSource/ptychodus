"""Guards for the settings-controlled chat endpoint."""

from urllib.parse import urlparse
import logging
import os

logger = logging.getLogger(__name__)

__all__ = ['get_api_key_for_base_url']

_LOOPBACK_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})


def get_api_key_for_base_url(base_url: str) -> str:
    """Return ``OPENAI_API_KEY``, or an empty string when the endpoint is not confidential.

    ``Agent/BaseURL`` is a plain string in ``settings.ini``, and batch mode loads that file
    from its input directory. Withholding the key from cleartext endpoints keeps a settings
    file that travels with a dataset from redirecting the user's credential to an attacker.
    """
    api_key = os.environ.get('OPENAI_API_KEY', '')

    if not api_key:
        return ''

    url = urlparse(base_url)

    if url.scheme == 'https':
        return api_key

    if url.scheme == 'http' and url.hostname in _LOOPBACK_HOSTS:
        return api_key

    logger.warning(
        f'Withholding OPENAI_API_KEY from insecure chat endpoint "{base_url}"; '
        'use an https:// URL to send credentials.'
    )
    return ''

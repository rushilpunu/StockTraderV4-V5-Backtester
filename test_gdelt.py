#!/usr/bin/env python3
"""Simple GDELT API connectivity test."""

import asyncio
import aiohttp
from loguru import logger

async def test_gdelt_direct():
    """Test GDELT API directly without our wrapper."""
    
    # Test different endpoints
    endpoints = [
        "https://api.gdeltproject.org/api/v2/doc/doc",
        "https://api.gdeltproject.org/api/v1/doc/doc",
        "https://api.gdeltproject.org/api/v2/doc/doc",
    ]
    
    test_params = [
        {"query": "Apple", "startdatetime": "20250101000000", "enddatetime": "20250102000000", "maxrecords": 5},
        {"query": "Apple", "start": "20250101000000", "end": "20250102000000", "max": 5},
        {"query": "Apple", "startdatetime": "20250101000000", "enddatetime": "20250102000000", "maxrecords": 5},
    ]
    
    for i, (endpoint, params) in enumerate(zip(endpoints, test_params)):
        logger.info(f"Testing endpoint {i+1}: {endpoint}")
        logger.info(f"Params: {params}")
        
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint, params=params) as response:
                    logger.info(f"Status: {response.status}")
                    logger.info(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
                    
                    if response.status == 200:
                        text = await response.text()
                        logger.info(f"Response length: {len(text)} characters")
                        logger.info(f"First 200 chars: {text[:200]}")
                        
                        if len(text) > 0:
                            logger.success(f"✅ Endpoint {i+1} working - got {len(text)} characters")
                        else:
                            logger.warning(f"⚠️  Endpoint {i+1} returned empty response")
                    else:
                        logger.error(f"❌ Endpoint {i+1} failed with status {response.status}")
                        
        except asyncio.TimeoutError:
            logger.error(f"❌ Endpoint {i+1} timed out after 60 seconds")
        except Exception as e:
            logger.error(f"❌ Endpoint {i+1} error: {e}")
        
        logger.info("-" * 50)

if __name__ == "__main__":
    logger.info("Testing GDELT API connectivity directly...")
    asyncio.run(test_gdelt_direct())

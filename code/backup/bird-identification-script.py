#!/usr/bin/env python3
"""
Bird identification script for Lightroom XMP sidecar files.
Uses Qwen3.5 via vllm API to analyze images and update XMP files with bird identifications.
"""

import os
import re
import json
import time
import requests
from pathlib import Path
import base64
from xml.dom import minidom
import xml.etree.ElementTree as ET

# Configuration
XMP_2025_PATH = "/home/robin/.openclaw/workspace/data/lr-proj/Photos-2025"
XMP_2024_PATH = "/home/robin/.openclaw/workspace/data/lr-proj/Photos-2024"
JPG_PATH = "/home/robin/.openclaw/workspace/data/lr-proj/jpg"
REPORT_PATH = "/home/robin/.openclaw/workspace/bird-identification-report.json"

# vllm API configuration
VLLM_API_URL = "http://darwin:8080/v1/chat/completions"
VLLM_MODEL = "Qwen3.5"

def get_xmp_files(base_path):
    """Get all XMP files from the given directory recursively."""
    xmp_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('.xmp'):
                xmp_files.append(os.path.join(root, f))
    return sorted(xmp_files)

def extract_base_name(xmp_path):
    """Extract base name from XMP file (without extension)."""
    return os.path.splitext(os.path.basename(xmp_path))[0]

def find_corresponding_jpg(base_name):
    """Find the corresponding JPG file by base name."""
    jpg_path = os.path.join(JPG_PATH, f"{base_name}.jpg")
    if os.path.exists(jpg_path):
        return jpg_path
    
    # Try case-insensitive match
    try:
        for f in os.listdir(JPG_PATH):
            if f.lower() == base_name.lower() + '.jpg':
                return os.path.join(JPG_PATH, f)
    except:
        pass
    
    return None

def read_first_n_bytes(image_path, max_bytes=2*1024*1024):
    """Read first N bytes of image file."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()[:max_bytes]).decode('utf-8')

def analyze_image_with_vllm(image_path, retry_count=0):
    """Analyze image using Qwen3.5 model to identify birds."""
    try:
        # Read and encode image (limit to 2MB to avoid timeout)
        base64_image = read_first_n_bytes(image_path, 2*1024*1024)
        
        prompt = """Analyze this image carefully. Are there any birds visible?

If you see birds:
1. Identify the species name in English
2. If you know the Chinese name, provide it
3. Be specific about the species

Respond in this EXACT format ONLY:
BIRD: <english_name> | <chinese_name>

Example: BIRD: scaly-sided merganser | 中华秋沙鸭

If you don't see any birds, respond with exactly: NO_BIRDS

Be confident in your identification. If you're not sure it's a bird, say NO_BIRDS."""

        payload = {
            "model": VLLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(VLLM_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        print(f"  Model response: {content[:150]}...")
        
        # Parse the response
        if content.upper().startswith('BIRD:') or content.startswith('BIRD:'):
            # Extract bird info
            bird_part = content[5:].strip()
            if '|' in bird_part:
                parts = bird_part.split('|')
                english = parts[0].strip().lower()
                chinese = parts[1].strip().lower() if len(parts) > 1 else ""
                return english, chinese
            else:
                return bird_part.lower(), ""
        elif content.upper().startswith('NO_BIRDS') or 'no birds' in content.lower():
            return None, "No birds detected"
        else:
            # Try to extract bird info from any response
            if any(word in content.lower() for word in ['bird', 'duck', 'eagle', 'hawk', 'sparrow', 'pigeon', 'owl', 'swan', 'crane', 'heron', 'duck', 'goose', 'merganser', 'grebe', 'petrel', 'albatross', 'gull', 'tern', 'skua', 'auk', 'penguin', 'flamingo', 'stork', 'ibis', 'spoonbill', 'falcon', 'kite', 'harrier', 'buzzard', 'vulture', 'jay', 'crow', 'raven', 'magpie', 'starling', 'thrush', 'robin', 'wren', 'warbler', 'finch', 'bunting', 'lark', 'swallow', 'martin', 'swift', 'hummingbird', 'woodpecker', 'cuckoo', 'nightjar', 'swiftlet', 'kingfisher', 'hoopoe', 'hornbill', 'trogan', 'cormorant', 'darter', 'pelican', 'frigatebird', 'booby', 'gannet', 'tropicbird', 'shearwater', 'petrel']):
                return content.strip().lower(), ""
            return None, f"Could not parse response: {content}"
            
    except requests.exceptions.RequestException as e:
        if retry_count < 3:
            print(f"  API error, retrying... ({retry_count + 1}/3)")
            time.sleep(2)
            return analyze_image_with_vllm(image_path, retry_count + 1)
        return None, f"API error: {str(e)}"
    except Exception as e:
        return None, str(e)

def get_pinyin_initial(chinese_name):
    """Get the first letter of Chinese name's pinyin."""
    pinyin_map = {
        '中': 'z',  # zhong
        '白': 'b',  # bai
        '黑': 'h',  # hei
        '红': 'h',  # hong
        '绿': 'l',  # lv
        '蓝': 'l',  # lan
        '黄': 'h',  # huang
        '灰': 'h',  # hui
        '金': 'j',  # jin
        '银': 'y',  # yin
        '小': 'x',  # xiao
        '大': 'd',  # da
        '长': 'c',  # chang
        '短': 'd',  # duan
        '冠': 'g',  # guan
        '凤': 'f',  # feng
        '鹰': 'y',  # ying
        '隼': 's',  # sun
        '鸮': 'x',  # xiao
        '鹛': 'm',  # mei
        '雀': 'q',  # que
        '鸭': 'y',  # ya
        '鹅': 'e',  # e
        '雁': 'y',  # yan
        '鹤': 'h',  # he
        '鹳': 'g',  # guan
        '鹭': 'l',  # lu
        '鸥': 'o',  # ou
        '鸬': 'l',  # lu
        '鹚': 'c',  # ci
        '鸨': 'b',  # bao
        '鸻': 'h',  # heng
        '鹬': 'y',  # yu
        '鷸': 'y',  # yu
        '雕': 'd',  # diao
        '鵟': 'k',  # kuang
        '鹞': 'y',  # yao
        '鸢': 'y',  # yuan
        '鹏': 'p',  # peng
        '鸵': 't',  # tuo
        '鴯': 'e',  # er
        '鹂': 'l',  # li
        '画': 'h',  # hua
        '眉': 'm',  # mei
        '柳': 'l',  # liu
        '莺': 'y',  # ying
        '啄': 'z',  # zhuo
        '木': 'm',  # mu
        '鸟': 'n',  # niao
        '鸠': 'j',  # jiu
        '鸽': 'g',  # ge
        '斑': 'b',  # ban
        '鹃': 'd',  # juan
        '布': 'b',  # bu
        '鸺': 'x',  # xiu
        '鹗': 'e',  # e
        '鱼': 'y',  # yu
        '鸰': 'l',  # ling
        '鹡': 'j',  # ji
        '鹀': 'w',  # wu
        '鹨': 'l',  # liu
        '鹟': 'w',  # weng
        '鸫': 'd',  # dong
        '鸩': 'z',  # zhen
        '鸪': 'g',  # gu
        '鸩': 'z',  # zhen
        '鸲': 'q',  # qu
        '鹑': 'q',  # qun
        '鹖': 'h',  # he
        '鹗': 'e',  # e
        '鹆': 'yu',  # yu
        '鹇': 'x',  # xian
        '鹈': 't',  # ti
        '鹉': 'ren',  # ren
        '鹃': 'juan',  # juan
        '鵐': 'shi',  # shi
        '鵙': 'ju',  # ju
        '鵓': 'bo',  # bo
        '鵣': 'ai',  # ai
        '鵤': 'wu',  # wu
        '鵪': 'an',  # an
        '鵯': 'po',  # po
        '鵺': 'ye',  # ye
        '鶚': 'e',  # e
        '鶇': 'dong',  # dong
        '鶈': 'qi',  # qi
        '鹜': 'wu',  # wu
        '鹚': 'ci',  # ci
        '鹛': 'mei',  # mei
        '鶉': 'chun',  # chun
        '鶋': 'ju',  # ju
        '鿆': 'shi',  # shi
        '鶒': 'chi',  # chi
        '鶓': 'mao',  # mao
        '鶄': 'jing',  # jing
        '鶎': 'yuan',  # yuan
        '鶕': 'hang',  # hang
        '鹺': 'cuo',  # cuo
        '鹻': 'huan',  # huan
        '鹿': 'guan',  # guan
        '鸀': 'zhu',  # zhu
        '鸁': 'luo',  # luo
        '鸂': 'xi',  # xi
        '鸃': 'yi',  # yi
        '鸄': 'ju',  # ju
        '鸅': 'zha',  # zha
        '鸆': 'bo',  # bo
        '鸇': 'zheng',  # zheng
        '鸈': 'hu',  # hu
        '鸉': 'hua',  # hua
        '鸊': 'pi',  # pi
        '鸋': 'ni',  # ni
        '鸌': 'tao',  # tao
        '鸍': 'shi',  # shi
        '鸎': 'yuan',  # yuan
        '鸏': 'meng',  # meng
        '鸐': 'di',  # di
        '鸑': 'yue',  # yue
        '鸒': 'yu',  # yu
        '鸓': 'shu',  # shu
        '鸔': 'yan',  # yan
        '鸕': 'lu',  # lu
        '鸖': 'ting',  # ting
        '鸗': 'liu',  # liu
        '鸘': 'xiang',  # xiang
        '鸙': 'yao',  # yao
        '鸚': 'ying',  # ying
        '鸛': 'guan',  # guan
        '鸜': 'qu',  # qu
        '鸝': 'li',  # li
        '鸞': 'luan',  # luan
    }
    
    if chinese_name and len(chinese_name) > 0:
        first_char = chinese_name[0]
        return pinyin_map.get(first_char, first_char).lower()
    return 'x'  # default

def format_bird_subject(english_name, chinese_name):
    """Format bird subject according to specification: xyz-chinese_name-english_name"""
    if chinese_name:
        pinyin_initial = get_pinyin_initial(chinese_name)
        return f"{pinyin_initial}-{chinese_name.lower()}-{english_name.lower()}"
    else:
        # If no Chinese name, just use english with 'e' prefix
        return f"e-{english_name.lower()}"

def update_xmp_with_subject(xmp_path, bird_subject):
    """Update XMP file with bird subject information."""
    try:
        # Read the XMP file
        with open(xmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse XML
        root = ET.fromstring(content)
        
        # Find the main Description element
        rdf_ns = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
        dc_ns = '{http://purl.org/dc/elements/1.1/}'
        
        # Find Description element
        description = root.find(f'.//{rdf_ns}Description')
        if description is None:
            return False, "Could not find Description element"
        
        # Check if dc:subject already exists
        subject_elem = description.find(f'{dc_ns}subject')
        
        if subject_elem is None:
            # Create new dc:subject element
            ET.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
            subject_elem = ET.SubElement(description, f'{{{dc_ns[1:-1]}}}subject')
        
        # Get existing subjects if any
        existing_subjects = []
        if subject_elem.text:
            existing_subjects = [s.strip() for s in subject_elem.text.split(',') if s.strip()]
        
        # Add new bird info if not duplicate
        if bird_subject not in existing_subjects:
            all_subjects = existing_subjects + [bird_subject]
            subject_elem.text = ', '.join(all_subjects)
        
        # Write back with proper formatting
        tree = ET.ElementTree(root)
        
        # Convert to string for proper XML declaration
        xml_str = ET.tostring(root, encoding='utf-8')
        
        # Parse with minidom for pretty printing
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(encoding='utf-8')
        
        # Write back
        with open(xmp_path, 'wb') as f:
            f.write(pretty_xml)
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def main():
    """Main processing function."""
    results = {
        'processed_2025': [],
        'processed_2024': [],
        'errors': [],
        'no_birds': [],
        'summary': {
            'total_2025': 0,
            'total_2024': 0,
            'birds_found': 0,
            'no_birds_count': 0,
            'errors_count': 0
        }
    }
    
    # Process Photos-2025 first
    print("=" * 60)
    print("Processing Photos-2025...")
    print("=" * 60)
    xmp_files_2025 = get_xmp_files(XMP_2025_PATH)
    results['summary']['total_2025'] = len(xmp_files_2025)
    print(f"Found {len(xmp_files_2025)} XMP files in Photos-2025")
    
    for i, xmp_path in enumerate(xmp_files_2025):
        base_name = extract_base_name(xmp_path)
        jpg_path = find_corresponding_jpg(base_name)
        
        if jpg_path is None:
            error_msg = f"No matching JPG found for base name: {base_name}"
            results['errors'].append({
                'file': xmp_path,
                'error': error_msg
            })
            results['summary']['errors_count'] += 1
            print(f"[{i+1}/{len(xmp_files_2025)}] ERROR: {error_msg}")
            continue
        
        print(f"[{i+1}/{len(xmp_files_2025)}] Processing: {base_name}")
        
        # Analyze image
        bird_english, bird_chinese = analyze_image_with_vllm(jpg_path)
        
        if bird_english is None and "No birds detected" in str(bird_chinese):
            # No birds found
            results['no_birds'].append({
                'xmp': xmp_path,
                'jpg': jpg_path,
                'reason': bird_chinese
            })
            results['summary']['no_birds_count'] += 1
            print(f"  -> No birds detected")
        elif bird_english:
            # Found birds
            bird_subject = format_bird_subject(bird_english, bird_chinese)
            success, update_error = update_xmp_with_subject(xmp_path, bird_subject)
            
            if success:
                results['processed_2025'].append({
                    'xmp': xmp_path,
                    'jpg': jpg_path,
                    'bird_english': bird_english,
                    'bird_chinese': bird_chinese,
                    'subject_format': bird_subject
                })
                results['summary']['birds_found'] += 1
                print(f"  -> Bird found: {bird_subject}")
            else:
                results['errors'].append({
                    'file': xmp_path,
                    'bird_english': bird_english,
                    'bird_chinese': bird_chinese,
                    'error': f"XMP update failed: {update_error}"
                })
                results['summary']['errors_count'] += 1
                print(f"  -> ERROR updating XMP: {update_error}")
        else:
            results['errors'].append({
                'file': xmp_path,
                'jpg': jpg_path,
                'error': bird_chinese if bird_chinese else "Unknown error"
            })
            results['summary']['errors_count'] += 1
            print(f"  -> ERROR: {bird_chinese}")
        
        # Progress update every 50 files
        if (i + 1) % 50 == 0:
            print(f"\n*** Progress 2025: {i + 1}/{len(xmp_files_2025)} files processed ***")
            print(f"    Birds found: {results['summary']['birds_found']}")
            print(f"    No birds: {results['summary']['no_birds_count']}")
            print(f"    Errors: {results['summary']['errors_count']}")
            print()
    
    # Process Photos-2024
    print("\n" + "=" * 60)
    print("Processing Photos-2024...")
    print("=" * 60)
    xmp_files_2024 = get_xmp_files(XMP_2024_PATH)
    results['summary']['total_2024'] = len(xmp_files_2024)
    print(f"Found {len(xmp_files_2024)} XMP files in Photos-2024")
    
    for i, xmp_path in enumerate(xmp_files_2024):
        base_name = extract_base_name(xmp_path)
        jpg_path = find_corresponding_jpg(base_name)
        
        if jpg_path is None:
            error_msg = f"No matching JPG found for base name: {base_name}"
            results['errors'].append({
                'file': xmp_path,
                'error': error_msg
            })
            results['summary']['errors_count'] += 1
            print(f"[{i+1}/{len(xmp_files_2024)}] ERROR: {error_msg}")
            continue
        
        print(f"[{i+1}/{len(xmp_files_2024)}] Processing: {base_name}")
        
        # Analyze image
        bird_english, bird_chinese = analyze_image_with_vllm(jpg_path)
        
        if bird_english is None and "No birds detected" in str(bird_chinese):
            # No birds found
            results['no_birds'].append({
                'xmp': xmp_path,
                'jpg': jpg_path,
                'reason': bird_chinese
            })
            results['summary']['no_birds_count'] += 1
            print(f"  -> No birds detected")
        elif bird_english:
            # Found birds
            bird_subject = format_bird_subject(bird_english, bird_chinese)
            success, update_error = update_xmp_with_subject(xmp_path, bird_subject)
            
            if success:
                results['processed_2024'].append({
                    'xmp': xmp_path,
                    'jpg': jpg_path,
                    'bird_english': bird_english,
                    'bird_chinese': bird_chinese,
                    'subject_format': bird_subject
                })
                results['summary']['birds_found'] += 1
                print(f"  -> Bird found: {bird_subject}")
            else:
                results['errors'].append({
                    'file': xmp_path,
                    'bird_english': bird_english,
                    'bird_chinese': bird_chinese,
                    'error': f"XMP update failed: {update_error}"
                })
                results['summary']['errors_count'] += 1
                print(f"  -> ERROR updating XMP: {update_error}")
        else:
            results['errors'].append({
                'file': xmp_path,
                'jpg': jpg_path,
                'error': bird_chinese if bird_chinese else "Unknown error"
            })
            results['summary']['errors_count'] += 1
            print(f"  -> ERROR: {bird_chinese}")
        
        # Progress update every 50 files
        if (i + 1) % 50 == 0:
            print(f"\n*** Progress 2024: {i + 1}/{len(xmp_files_2024)} files processed ***")
            print(f"    Birds found: {results['summary']['birds_found']}")
            print(f"    No birds: {results['summary']['no_birds_count']}")
            print(f"    Errors: {results['summary']['errors_count']}")
            print()
    
    # Save report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total 2025 files: {results['summary']['total_2025']}")
    print(f"Total 2024 files: {results['summary']['total_2024']}")
    print(f"Birds found: {results['summary']['birds_found']}")
    print(f"No birds: {results['summary']['no_birds_count']}")
    print(f"Errors: {results['summary']['errors_count']}")
    print(f"Report saved to: {REPORT_PATH}")

if __name__ == '__main__':
    main()

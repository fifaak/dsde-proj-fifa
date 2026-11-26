"""Extract selected truck driving restrictions and retain their scope and source."""

from datetime import date
from hashlib import sha256
from io import BytesIO
import re

from pypdf import PdfReader


OTP_PDF = "https://www.otp.go.th/uploads/tiny_uploads/PDF/2569-03/20260319_5_mot08081_planon_2569_Truck.pdf"
OTP_ROAD_REPORT = "https://www.otp.go.th/uploads/tiny_uploads/ProjectOTP/2557/Project07/01_Final_Report_PDF.pdf"
GAZETTE_PDF = "https://ratchakitcha.soc.go.th/documents/58726.pdf"
GAZETTE_SHA256 = "1e4b156bd546ae52aeb936fa995d45f75e8680c27a907d31b85e54ba4c72d645"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def restriction_rows(download):
    plan = PdfReader(BytesIO(download(OTP_PDF)))
    for index, page in enumerate(plan.pages):
        text = " ".join(page.extract_text().translate(THAI_DIGITS).split())
        area_rule = re.search(
            r"มาตรการระยะที่ 3 ห้ามรถบรรทุกตั้งแต่ (\d+) ล้อขึ้นไป เข้าในเขตพื้นที่ (\d+) ตารางกิโลเมตร"
            r" ในช่วงเวลา (\d{2}\.\d{2})\s*-\s*(\d{2}\.\d{2}) น\. "
            r"\(เริ่มบังคับใช้ (\d+) เมษายน (\d{4})\)",
            text,
        )
        if area_rule:
            break
    else:
        raise ValueError("The expected current Bangkok restriction was not found in the OTP PDF")

    # The Gazette PDF uses an embedded font whose body does not decode as Thai.
    # These clause fields were checked against the rendered PDF and are tied to that revision.
    if sha256(download(GAZETTE_PDF)).hexdigest() != GAZETTE_SHA256:
        raise ValueError("The Royal Gazette PDF changed. Review clauses 2 and 4–6 before regenerating.")

    wheels, area, start, end, day, buddhist_year = area_rule.groups()
    yield {
        "rule_id": "bkk-inner-ring-phase-3",
        "scope_type": "area",
        "road_name": "",
        "road_segment": f"เขตวงแหวนชั้นในรัชดาภิเษก–จรัญสนิทวงศ์ {area} ตร.กม.",
        "vehicle_type": f"รถบรรทุกตั้งแต่ {wheels} ล้อขึ้นไป รถบรรทุกตั้งแต่ 3 เพลาขึ้นไป และรถพ่วง",
        "days_of_week": "ทุกวัน เว้นวันหยุดราชการ",
        "start_time": start.replace(".", ":"),
        "end_time": end.replace(".", ":"),
        "direction": "",
        "exceptions": "รายงาน สนข. ระบุการผ่อนผัน 10:00–15:00 บนถนนวงรอบ และเงื่อนไขเฉพาะสำหรับบางช่วงถนน ด่านทางด่วน และสินค้าบางประเภท",
        "effective_from": date(int(buddhist_year) - 543, 4, int(day)).isoformat(),
        "effective_to": "",
        "source_type": "government_summary",
        "source_reference": "สนข. แผนปี 2569–2580 หัวข้อ 4.1.2 ระยะที่ 3 และรูป 4-3",
        "source_url": f"{OTP_PDF}#page={index + 1}",
        "supporting_source_url": OTP_ROAD_REPORT,
    }

    routes = (
        ("chaloem-maha-nakhon", "ทางพิเศษเฉลิมมหานคร", ""),
        ("si-rat", "ทางพิเศษศรีรัช", ""),
        ("chalong-rat", "ทางพิเศษฉลองรัช", ""),
        ("udon-ratthaya", "ทางพิเศษอุดรรัถยา", ""),
        ("prachim-ratthaya", "ทางพิเศษประจิมรัถยา", ""),
        ("rama-3", "ทางพิเศษสายพระราม 3–ดาวคะนอง–วงแหวนรอบนอกกรุงเทพมหานครด้านตะวันตก",
         "ด่านเก็บค่าผ่านทางพิเศษสุขสวัสดิ์ ถึงสะพานทศมราชัน (บางโคล่)"),
    )
    vehicle_windows = (
        (4, "รถบรรทุก 6 ล้อ", "06:00", "09:00"),
        (4, "รถบรรทุก 6 ล้อ", "16:00", "20:00"),
        (5, "รถบรรทุกตั้งแต่ 10 ล้อขึ้นไป และรถพ่วง", "06:00", "09:00"),
        (5, "รถบรรทุกตั้งแต่ 10 ล้อขึ้นไป และรถพ่วง", "15:00", "21:00"),
    )
    for shortcut, name, segment in routes:
        for clause, vehicle, start, end in vehicle_windows:
            yield {
                "rule_id": f"exat-2568-{shortcut}-clause-{clause}-{start.replace(':', '')}",
                "scope_type": "expressway_segment" if segment else "expressway",
                "road_name": name,
                "road_segment": segment,
                "vehicle_type": vehicle,
                "days_of_week": "ทุกวัน เว้นวันหยุดราชการ",
                "start_time": start,
                "end_time": end,
                "direction": "",
                "exceptions": "รถบรรทุก กทพ. รถตำรวจขณะปฏิบัติหน้าที่ และรถที่ได้รับอนุญาตตามข้อ 6",
                "effective_from": "2025-01-29",
                "effective_to": "",
                "source_type": "royal_gazette",
                "source_reference": f"ราชกิจจานุเบกษา เล่ม 142 ตอนพิเศษ 31 ง หน้า 22–23 ข้อ {clause} และ 6",
                "source_url": GAZETTE_PDF,
                "supporting_source_url": "",
            }

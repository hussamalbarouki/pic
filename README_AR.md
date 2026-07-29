# واجهة ComfyUI البسيطة للتعديل الموضعي

واجهة عربية خفيفة تعمل فوق ComfyUI وتستخدم Workflow الـInpainting المرفق.

## ما الذي توفره؟

- رفع الصورة الأصلية.
- رسم Mask بالماوس أو اللمس.
- تعديل Positive Prompt وNegative Prompt.
- تعديل Steps وCFG وDenoise وSeed وGrow Mask.
- اختيار Sampler وScheduler وCheckpoint من ComfyUI.
- تعديل Workflow JSON كاملاً، لذلك كل عقدة وكل قيمة قابلة للتعديل.
- شريط تقدم حي بالنسبة المئوية اعتماداً على WebSocket الخاص بـComfyUI.
- زر إيقاف.
- عرض النتيجة وزر تحميلها.
- تحديد حد أقصى لحجم الصورة لحماية VPS الذي يعمل على CPU.

## متطلبات التشغيل

- ComfyUI يعمل ويمكن الوصول إليه من الحاوية عبر `http://comfyui:8188`.
- النموذج `Realistic_Vision_V3.0-inpainting.safetensors` موجود في ComfyUI.
- يفضّل وضع `editor-web` داخل نفس Docker Compose ونفس شبكة ComfyUI.

## متغيرات البيئة

- `COMFY_URL`: رابط ComfyUI الداخلي. الافتراضي `http://comfyui:8188`.
- `APP_USERNAME` و`APP_PASSWORD`: حماية اختيارية بكلمة مرور. اتركهما فارغين لتعطيلها.
- `DATA_DIR`: مكان حفظ النتائج المؤقتة.
- `MAX_UPLOAD_MB`: الحد الأعلى لكل ملف.
- `JOB_RETENTION_HOURS`: حذف النتائج القديمة عند تشغيل الحاوية.

## تشغيل محلي للاختبار

```bash
cp docker-compose.example.yml docker-compose.yml
docker compose up -d --build
```

ثم افتح المنفذ 8000 عبر البروكسي. لا تفتح المنفذ مباشرة للعالم إذا كانت الصور خاصة.

## إضافتها إلى Compose الحالي في Coolify

الأفضل أن يكون هيكل الخدمات هكذا:

```yaml
services:
  comfyui:
    # خدمة ComfyUI الموجودة لديك

  editor-web:
    build:
      context: ./editor
      dockerfile: Dockerfile
    environment:
      COMFY_URL: http://comfyui:8188
      APP_USERNAME: ${EDITOR_USERNAME}
      APP_PASSWORD: ${EDITOR_PASSWORD}
    expose:
      - "8000"
    depends_on:
      - comfyui
```

في Coolify اربط الدومين بخدمة `editor-web` والمنفذ الداخلي `8000`، مثل:

```text
https://editor.YOUR_SERVER_IP.sslip.io:8000
```

يظل الاتصال بين الواجهة وComfyUI داخلياً عبر HTTP، بينما واجهة المستخدم الخارجية تكون HTTPS.

## ملاحظة شريط التقدم

الواجهة تتصل بواجهة WebSocket في ComfyUI وتقرأ رسائل `progress` (`value` و`max`). خلال تحميل النموذج قد يبقى التقدم عند 10% بعض الوقت، ثم يبدأ بالتحرك مع خطوات الـSampler.

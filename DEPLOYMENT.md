# Streamlit Cloud Deployment Guide

## 🚀 Deploy Tanpa Fork Button

Aplikasi ini sudah dikonfigurasi untuk deployment di Streamlit Cloud tanpa menampilkan Fork button atau GitHub link.

### Konfigurasi yang Sudah Diterapkan:

1. **`.streamlit/config.toml`** - Hide toolbar dan menu items
2. **Custom CSS** di `app.py` - Hide GitHub links dan Fork button
3. **Menu items** di `st.set_page_config()` - Semua diset ke `None`

### Cara Deploy:

#### Option 1: Streamlit Cloud (Recommended)

1. **Push repo ke GitHub**
   ```bash
   git add .
   git commit -m "Clean deployment ready"
   git push origin main
   ```

2. **Deploy di Streamlit Cloud**
   - Buka https://share.streamlit.io/
   - Login dengan GitHub
   - Click "New app"
   - Pilih repository: `hub`
   - Main file path: `src/app.py`
   - Click "Deploy"

3. **Verifikasi**
   - Setelah deploy selesai, cek apakah Fork button hilang
   - Jika masih ada, coba refresh browser dengan Ctrl+F5

#### Option 2: Docker (Self-hosted)

```bash
# Build image
docker build -t f1-analytics .

# Run container
docker run -p 8501:8501 f1-analytics
```

Access: http://localhost:8501

#### Option 3: Local Development

```bash
# Activate venv
.venv\Scripts\activate

# Run app
streamlit run src/app.py
```

### 🔧 Troubleshooting

**Q: Fork button masih muncul?**
A: 
- Clear browser cache (Ctrl+Shift+Delete)
- Hard refresh (Ctrl+F5)
- Check `.streamlit/config.toml` sudah di-push

**Q: App tidak bisa dibuka?**
A:
- Check requirements.txt lengkap
- Pastikan `f1_cache/` folder ada
- Check logs di Streamlit Cloud dashboard

### 📝 Notes:

- **Cache**: FastF1 cache akan di-generate otomatis saat pertama kali load
- **Data**: CSV files di `data/` folder akan di-load otomatis
- **Performance**: First load bisa lambat karena download F1 data

### ✨ Features yang Sudah Dirapikan:

✅ Hide Fork button  
✅ Hide GitHub link  
✅ Hide menu items (Get Help, Report Bug, About)  
✅ Hide Streamlit branding  
✅ Clean professional look  
✅ Dark theme dengan F1 colors  

---
**Created by Maxvy** 🏎️

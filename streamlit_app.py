import streamlit as st
import whisper
import tempfile
import os
import json
import gc
import torch
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="🤖 AI音声文字起こしツール",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 元のindex.htmlと同じデザインのCSS
st.markdown("""
<style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #ff6b6b 100%);
        min-height: 100vh;
    }

    .main .block-container {
        max-width: 1400px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }

    .title {
        text-align: center;
        color: #333;
        margin-bottom: 15px;
        font-size: 2.8em;
        font-weight: 300;
        background: linear-gradient(45deg, #667eea, #764ba2, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
        font-size: 1.3em;
    }

    .ai-badge {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }

    .badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        color: white;
    }

    .badge-ai { background: linear-gradient(45deg, #ff6b6b, #ff8e8e); }
    .badge-ml { background: linear-gradient(45deg, #4ecdc4, #44a08d); }
    .badge-dl { background: linear-gradient(45deg, #667eea, #764ba2); }
    .badge-realtime { background: linear-gradient(45deg, #feca57, #ff9ff3); }

    .upload-section {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 30px;
        border: 2px dashed #dee2e6;
    }

    .result-output {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 25px;
        min-height: 200px;
        font-family: 'Noto Sans JP', sans-serif;
        font-size: 1.1em;
        line-height: 1.8;
        color: #333;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }

    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 25px !important;
        font-size: 1em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3) !important;
    }

    .memory-warning {
        background: linear-gradient(90deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }

    .model-info {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# メモリ効率的なモデル管理
class WhisperModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_size = None
        self.model_info = {
            "tiny": {"size": "39MB", "speed": "最高速", "accuracy": "基本"},
            "base": {"size": "74MB", "speed": "高速", "accuracy": "良好"},
            "small": {"size": "244MB", "speed": "中速", "accuracy": "高精度"},
            "medium": {"size": "769MB", "speed": "低速", "accuracy": "より高精度"},
        }
    
    def cleanup_memory(self):
        """メモリクリーンアップ"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_size = None
        
        # GPU/CPUメモリクリーンアップ
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Python ガベージコレクション
        gc.collect()
    
    def load_model_safely(self, model_size):
        """安全なモデル読み込み"""
        try:
            # モデルが既に読み込まれていて同じサイズの場合はそのまま使用
            if self.current_model is not None and self.current_model_size == model_size:
                return self.current_model
            
            # メモリ制限チェック
            if model_size in ["large"]:
                st.error("❌ largeモデルはStreamlit Cloudのメモリ制限により利用できません")
                return None
            
            # 既存モデルをクリーンアップ
            if self.current_model is not None:
                st.info("🔄 前のモデルをメモリから解放中...")
                self.cleanup_memory()
            
            # 新しいモデルを読み込み
            with st.spinner(f"🤖 {model_size}モデル読み込み中... ({self.model_info[model_size]['size']})"):
                self.current_model = whisper.load_model(model_size)
                self.current_model_size = model_size
            
            # 成功メッセージ
            info = self.model_info[model_size]
            st.success(f"✅ {model_size}モデル読み込み完了！ ({info['size']}, {info['speed']}, {info['accuracy']})")
            
            return self.current_model
            
        except Exception as e:
            st.error(f"❌ モデル読み込みエラー: {str(e)}")
            self.cleanup_memory()
            return None
    
    def get_model_info(self, model_size):
        """モデル情報を取得"""
        return self.model_info.get(model_size, {})

# グローバルモデルマネージャー
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = WhisperModelManager()

def format_time(seconds):
    """秒を分:秒形式に変換"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def safe_file_extension(filename):
    """安全なファイル拡張子を取得"""
    if not filename:
        return ".wav"
    ext = os.path.splitext(filename)[1].lower()
    supported_exts = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    return ext if ext in supported_exts else ".wav"

def transcribe_audio(audio_file, model_size, language, enable_timestamps, is_recording=False):
    """音声ファイルを文字起こし（メモリ効率版）"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: モデル読み込み（安全版）
        status_text.text("🤖 AIモデル準備中...")
        progress_bar.progress(20)
        
        model = st.session_state.model_manager.load_model_safely(model_size)
        if model is None:
            return None, None
        
        # Step 2: 一時ファイル作成
        status_text.text("📁 音声ファイル準備中...")
        progress_bar.progress(40)
        
        file_extension = ".wav" if is_recording else safe_file_extension(audio_file.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            try:
                file_content = audio_file.getvalue() if hasattr(audio_file, 'getvalue') else audio_file.read()
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            except Exception as e:
                st.error(f"ファイル書き込みエラー: {e}")
                return None, None
        
        # Step 3: 文字起こし設定
        status_text.text("⚙️ AI解析設定中...")
        progress_bar.progress(60)
        
        # メモリ効率的なWhisperオプション
        options = {
            "language": None if language == "auto" else language,
            "verbose": False,
            "fp16": False,  # メモリ使用量削減
        }
        
        if enable_timestamps:
            options["word_timestamps"] = True
        
        # Step 4: AI解析実行
        status_text.text("🔍 AI音声解析中...")
        progress_bar.progress(80)
        
        start_time = datetime.now()
        
        try:
            result = model.transcribe(tmp_file_path, **options)
        except Exception as whisper_error:
            st.error(f"❌ 音声解析エラー: {str(whisper_error)}")
            return None, None
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Step 5: 結果整理
        status_text.text("📝 結果を整理中...")
        progress_bar.progress(100)
        
        transcribed_text = result.get("text", "").strip()
        if not transcribed_text:
            st.warning("⚠️ 音声から文字を検出できませんでした")
            return None, None
        
        # 結果データ作成
        transcription_result = {
            "text": transcribed_text,
            "language": result.get("language", "unknown"),
            "processing_time": processing_time,
            "model_used": model_size,
            "char_count": len(transcribed_text),
            "word_count": len(transcribed_text.split()),
            "timestamp": datetime.now().isoformat(),
            "confidence": 1.0 - result.get("no_speech_prob", 0.0)
        }
        
        # セグメント情報
        segments = None
        if enable_timestamps and "segments" in result:
            segments = result["segments"]
        
        # 一時ファイル削除
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        
        # UI要素をクリア
        progress_bar.empty()
        status_text.empty()
        
        # 成功メッセージ
        st.success(f"🎉 文字起こし完了！ 処理時間: {processing_time:.2f}秒")
        
        return transcription_result, segments
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ 処理エラー: {str(e)}")
        
        # エラー時のクリーンアップ
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass
        
        return None, None

def display_model_info(model_size):
    """モデル情報を表示"""
    if model_size in st.session_state.model_manager.model_info:
        info = st.session_state.model_manager.model_info[model_size]
        st.markdown(f"""
        <div class="model-info">
            <strong>📊 選択中のモデル: {model_size.upper()}</strong><br>
            📦 サイズ: {info['size']} | ⚡ 速度: {info['speed']} | 🎯 精度: {info['accuracy']}
        </div>
        """, unsafe_allow_html=True)

def display_memory_warning():
    """メモリ使用量の警告を表示"""
    st.markdown("""
    <div class="memory-warning">
        ⚠️ メモリ効率化のため、モデル切り替え時に前のモデルを自動解放します<br>
        💡 安定動作のため「base」モデルを推奨します
    </div>
    """, unsafe_allow_html=True)

def main():
    # タイトル
    st.markdown("""
    <h1 class="title">🤖 AI音声文字起こしツール</h1>
    <div class="subtitle">安定版 - メモリ効率化対応</div>
    """, unsafe_allow_html=True)

    # AI機能バッジ
    st.markdown("""
    <div class="ai-badge">
        <div class="badge badge-ai">🧠 適応学習AI</div>
        <div class="badge badge-ml">📊 機械学習分析</div>
        <div class="badge badge-dl">🔬 深層学習処理</div>
        <div class="badge badge-realtime">⚡ リアルタイム解析</div>
    </div>
    """, unsafe_allow_html=True)

    # メモリ警告
    display_memory_warning()

    # サイドバー設定
    with st.sidebar:
        st.markdown("## ⚙️ 設定パネル")
        
        st.markdown("### 🤖 AIモデル選択")
        
        # モデル選択（安全版）
        model_size = st.selectbox(
            "処理速度と精度のバランス",
            options=["tiny", "base", "small", "medium"],  # largeを除外
            index=1,  # baseをデフォルト
            help="安定動作のため「base」を推奨します",
            key="model_selector"
        )
        
        # モデル情報表示
        display_model_info(model_size)
        
        # モデル切り替え警告
        if st.session_state.model_manager.current_model_size and st.session_state.model_manager.current_model_size != model_size:
            st.warning(f"⚠️ モデルを{st.session_state.model_manager.current_model_size}から{model_size}に変更します")
        
        st.markdown("### 🌍 言語設定")
        language = st.selectbox(
            "認識する言語",
            options=["auto", "ja", "en", "zh", "ko", "es", "fr", "de", "ru"],
            index=0,
            format_func=lambda x: {
                "auto": "🤖 自動検出（推奨）",
                "ja": "🇯🇵 日本語",
                "en": "🇺🇸 English",
                "zh": "🇨🇳 中文",
                "ko": "🇰🇷 한국어",
                "es": "🇪🇸 Español",
                "fr": "🇫🇷 Français",
                "de": "🇩🇪 Deutsch",
                "ru": "🇷🇺 Русский"
            }.get(x, x)
        )
        
        st.markdown("### ⏰ 詳細設定")
        enable_timestamps = st.checkbox(
            "タイムスタンプを有効にする", 
            value=True,
            help="音声の時間区切り情報を取得します"
        )
        
        # メモリクリーンアップボタン
        st.markdown("### 🧹 メモリ管理")
        if st.button("🗑️ メモリクリーンアップ", help="モデルをメモリから解放します"):
            st.session_state.model_manager.cleanup_memory()
            st.success("✅ メモリをクリーンアップしました")
            st.rerun()
        
        # 現在のメモリ状態表示
        if st.session_state.model_manager.current_model_size:
            st.info(f"📦 読み込み済み: {st.session_state.model_manager.current_model_size}モデル")
        else:
            st.info("📦 モデル未読み込み")

    # メインコンテンツ
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📁 ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "音声ファイルを選択",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac'],
            help="WAV, MP3, M4A形式を推奨（25MB以下）"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            if file_size > 25:
                st.error("❌ ファイルサイズが25MBを超えています")
            else:
                st.success(f"✅ ファイル選択済み: {uploaded_file.name} ({file_size:.1f}MB)")
                
                # 音声プレビュー
                if uploaded_file.type.startswith('audio/'):
                    st.audio(uploaded_file.getvalue())
        
        # 文字起こし実行ボタン
        if st.button("🚀 文字起こし開始", type="primary", use_container_width=True):
            if uploaded_file is not None:
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size <= 25:
                    result, segments = transcribe_audio(uploaded_file, model_size, language, enable_timestamps)
                    if result:
                        st.session_state['result'] = result
                        st.session_state['segments'] = segments
                        st.rerun()
                else:
                    st.error("❌ ファイルサイズが25MBを超えています")
            else:
                st.error("❌ 音声ファイルを選択してください")

    with col2:
        st.markdown("## 🎤 リアルタイム録音")
        
        st.info("💡 マイク録音機能（メモリ効率化版）")
        
        audio_value = st.audio_input("🎙️ 録音ボタンを押して話しかけてください")
        
        if audio_value is not None:
            st.success("✅ 録音完了！")
            
            if st.button("🔍 録音音声を文字起こし", use_container_width=True, type="secondary"):
                result, segments = transcribe_audio(audio_value, model_size, language, enable_timestamps, is_recording=True)
                if result:
                    st.session_state['result'] = result
                    st.session_state['segments'] = segments
                    st.rerun()

    # 結果表示エリア
    st.markdown("---")
    
    if 'result' in st.session_state and st.session_state['result']:
        result = st.session_state['result']
        segments = st.session_state.get('segments')
        
        # 統計情報
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 文字数", result['char_count'])
        with col2:
            st.metric("📝 単語数", result['word_count'])
        with col3:
            st.metric("⏱️ 処理時間", f"{result['processing_time']:.2f}秒")
        with col4:
            st.metric("🌍 検出言語", result['language'].upper())
        
        # タブ表示
        tab1, tab2 = st.tabs(["📝 文字起こし結果", "⏰ タイムスタンプ付き"])
        
        with tab1:
            st.markdown("### 📝 文字起こし結果")
            st.markdown(f"""
            <div class="result-output">
                {result['text']}
            </div>
            """, unsafe_allow_html=True)
            
            # ダウンロード
            st.download_button(
                "💾 テキストをダウンロード",
                data=result['text'],
                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with tab2:
            if segments and enable_timestamps:
                st.markdown("### ⏰ タイムスタンプ付きセグメント")
                
                for segment in segments:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    
                    start_formatted = format_time(start_time)
                    end_formatted = format_time(end_time)
                    
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #667eea;">
                        <strong>[{start_formatted} - {end_formatted}]</strong><br>
                        {text}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("タイムスタンプ機能を有効にして文字起こしを実行してください")
    
    else:
        st.markdown("## 📝 結果表示エリア")
        st.info("音声ファイルをアップロードまたは録音して、文字起こしを開始してください")
    
    # クリアボタン
    if st.button("🗑️ 全てクリア", use_container_width=True):
        # セッション状態をクリア
        for key in ['result', 'segments']:
            if key in st.session_state:
                del st.session_state[key]
        
        # メモリクリーンアップ
        st.session_state.model_manager.cleanup_memory()
        st.success("✅ 全てクリアしました")
        st.rerun()

if __name__ == "__main__":
    main()

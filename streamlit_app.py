import streamlit as st
import whisper
import tempfile
import os
import json
from datetime import datetime
import logging

# ページ設定
st.set_page_config(
    page_title="🤖 AI音声文字起こしツール",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS - 美しいデザイン
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .result-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        min-height: 200px;
        font-family: 'Noto Sans JP', sans-serif;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .success-banner {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Whisperモデルをキャッシュ（重要：メモリ効率化）
@st.cache_resource
def load_whisper_model(model_size):
    """Whisperモデルを読み込み（キャッシュ付き）"""
    try:
        with st.spinner(f"🤖 {model_size}モデルを読み込み中..."):
            model = whisper.load_model(model_size)
        st.success(f"✅ {model_size}モデル読み込み完了！")
        return model
    except Exception as e:
        st.error(f"❌ モデル読み込みエラー: {e}")
        return None

def format_time(seconds):
    """秒を分:秒形式に変換"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def transcribe_audio(audio_file, model_size, language, enable_timestamps, is_recording=False):
    """音声ファイルを文字起こし"""
    
    # プログレスバー表示
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: モデル読み込み
        status_text.text("🤖 AIモデル準備中...")
        progress_bar.progress(20)
        
        model = load_whisper_model(model_size)
        if model is None:
            st.error("❌ Whisperモデルの読み込みに失敗しました")
            return None
        
        # Step 2: 一時ファイル作成
        status_text.text("📁 音声ファイル準備中...")
        progress_bar.progress(40)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            if is_recording:
                tmp_file.write(audio_file.getvalue())
            else:
                tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Step 3: 文字起こし設定
        status_text.text("⚙️ AI解析設定中...")
        progress_bar.progress(60)
        
        # Whisperオプション設定
        options = {
            "language": None if language == "auto" else language,
            "verbose": False,
        }
        
        if enable_timestamps:
            options["word_timestamps"] = True
        
        # Step 4: AI解析実行
        status_text.text("🔍 AI音声解析中... (この処理には少し時間がかかります)")
        progress_bar.progress(80)
        
        start_time = datetime.now()
        result = model.transcribe(tmp_file_path, **options)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Step 5: 結果整理
        status_text.text("📝 結果を整理中...")
        progress_bar.progress(100)
        
        # 結果データ作成
        transcription_result = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "processing_time": processing_time,
            "model_used": model_size,
            "char_count": len(result["text"].strip()),
            "word_count": len(result["text"].strip().split()),
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
        
        # プログレスバー完了
        progress_bar.empty()
        status_text.empty()
        
        # 成功メッセージ
        st.markdown(f"""
        <div class="success-banner">
            🎉 文字起こし完了！ 処理時間: {processing_time:.2f}秒
        </div>
        """, unsafe_allow_html=True)
        
        return transcription_result, segments
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ 文字起こしエラー: {str(e)}")
        
        # 一時ファイル削除（エラー時）
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass
        
        return None, None

def display_results(result, segments):
    """結果を美しく表示"""
    
    # 統計情報をカード形式で表示
    st.markdown("### 📊 解析結果統計")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result['char_count']}</h3>
            <p>文字数</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result['word_count']}</h3>
            <p>単語数</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result['processing_time']:.1f}秒</h3>
            <p>処理時間</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result['language'].upper()}</h3>
            <p>検出言語</p>
        </div>
        """, unsafe_allow_html=True)
    
    # タブで結果を整理表示
    tab1, tab2, tab3 = st.tabs(["📝 文字起こし結果", "⏰ タイムスタンプ付き", "📄 詳細レポート"])
    
    with tab1:
        st.markdown("### 📝 文字起こし結果")
        st.markdown(f"""
        <div class="result-box">
            {result['text']}
        </div>
        """, unsafe_allow_html=True)
        
        # ダウンロードボタン
        st.download_button(
            label="💾 テキストファイルをダウンロード",
            data=result['text'],
            file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with tab2:
        if segments:
            st.markdown("### ⏰ タイムスタンプ付きセグメント")
            
            for i, segment in enumerate(segments):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                # 時間フォーマット
                start_formatted = format_time(start_time)
                end_formatted = format_time(end_time)
                
                st.markdown(f"""
                <div class="feature-card">
                    <strong>🕒 [{start_formatted} - {end_formatted}]</strong><br>
                    {text}
                </div>
                """, unsafe_allow_html=True)
            
            # セグメントデータのダウンロード
            segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
            st.download_button(
                label="💾 セグメントデータ（JSON）をダウンロード",
                data=segments_json,
                file_name=f"segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("💡 タイムスタンプ機能を有効にして文字起こしを実行すると、ここに時間区切りの詳細が表示されます")
    
    with tab3:
        st.markdown("### 📄 詳細レポート")
        
        report_data = f"""
📊 **音声解析レポート**

🕒 **処理情報**
- 処理開始時刻: {result['timestamp'][:19].replace('T', ' ')}
- 処理時間: {result['processing_time']:.2f}秒
- 使用AIモデル: {result['model_used']}
- 検出言語: {result['language']}

📝 **テキスト統計**
- 総文字数: {result['char_count']}文字
- 総単語数: {result['word_count']}語
- 平均的な信頼度: {result['confidence']:.1%}

🎯 **解析品質**
- モデル精度: {'高精度' if result['model_used'] in ['medium', 'large'] else '標準' if result['model_used'] == 'small' else '高速'}
- 音声品質: {'良好' if result['confidence'] > 0.8 else '普通' if result['confidence'] > 0.6 else '要改善'}
        """
        
        st.markdown(report_data)
        
        # 完全レポートのダウンロード
        full_report = generate_full_report(result, segments)
        st.download_button(
            label="📄 完全レポートをダウンロード",
            data=full_report,
            file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def generate_full_report(result, segments):
    """完全なレポートを生成"""
    report = f"""
AI音声文字起こし完全レポート
==========================================

📊 解析概要
処理時刻: {result['timestamp'][:19].replace('T', ' ')}
使用AIモデル: Whisper {result['model_used']}
検出言語: {result['language']}
処理時間: {result['processing_time']:.2f}秒
信頼度スコア: {result['confidence']:.1%}

📝 テキスト統計
総文字数: {result['char_count']}文字
総単語数: {result['word_count']}語
平均単語長: {result['char_count']/max(result['word_count'], 1):.1f}文字/語

📄 文字起こし結果
==========================================
{result['text']}

"""
    
    if segments:
        report += "\n⏰ タイムスタンプ付き詳細セグメント\n"
        report += "==========================================\n"
        
        for i, segment in enumerate(segments, 1):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            start_formatted = format_time(start_time)
            end_formatted = format_time(end_time)
            
            report += f"\n{i:3d}. [{start_formatted} - {end_formatted}] {text}"
        
        report += f"\n\n📈 セグメント統計: 総{len(segments)}セグメント\n"
    
    report += f"\n\n🤖 Generated by AI音声文字起こしツール\n"
    report += f"📅 レポート作成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n"
    
    return report

# メイン関数
def main():
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI音声文字起こしツール</h1>
        <p>OpenAI Whisper搭載 - 高精度音声認識システム</p>
        <p>🌟 多言語対応 | ⚡ 高速処理 | 📱 どこからでもアクセス可能</p>
    </div>
    """, unsafe_allow_html=True)

    # サイドバー設定
    with st.sidebar:
        st.markdown("## ⚙️ 設定パネル")
        
        st.markdown("### 🤖 AIモデル選択")
        model_size = st.selectbox(
            "処理速度と精度のバランス",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # baseをデフォルト
            help="""
            🚀 tiny: 最高速・基本精度 (39MB)
            ⚡ base: バランス型・推奨 (74MB)
            🎯 small: 高精度・中速 (244MB)
            🏆 medium: より高精度・低速 (769MB)
            👑 large: 最高精度・最低速 (1550MB)
            """
        )
        
        st.markdown("### 🌍 言語設定")
        language = st.selectbox(
            "認識する言語",
            options=["auto", "ja", "en", "zh", "ko", "es", "fr", "de", "ru"],
            index=0,  # 自動検出をデフォルト
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
        
        st.markdown("---")
        st.markdown("""
        ### 📚 機能一覧
        ✅ **高精度音声認識**  
        ✅ **9言語対応**  
        ✅ **タイムスタンプ表示**  
        ✅ **結果ダウンロード**  
        ✅ **リアルタイム録音**  
        ✅ **詳細レポート生成**  
        
        ### 📱 対応ファイル
        **音声**: MP3, WAV, M4A, FLAC, OGG, AAC  
        **動画**: MP4, AVI, MOV, MKV, WebM
        """)

    # メインコンテンツエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📁 ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "音声・動画ファイルを選択",
            type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="ドラッグ&ドロップまたはクリックしてファイルを選択"
        )
        
        if uploaded_file is not None:
            # ファイル情報を美しく表示
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            st.markdown(f"""
            <div class="feature-card">
                <h4>📄 選択されたファイル</h4>
                <p><strong>ファイル名:</strong> {uploaded_file.name}</p>
                <p><strong>サイズ:</strong> {file_size:.1f}MB</p>
                <p><strong>タイプ:</strong> {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 音声プレビュー（音声ファイルの場合）
            if uploaded_file.type.startswith('audio/'):
                st.audio(uploaded_file.getvalue())
        
        # 文字起こし実行ボタン
        if st.button("🚀 文字起こし開始", type="primary", use_container_width=True):
            if uploaded_file is not None:
                result, segments = transcribe_audio(uploaded_file, model_size, language, enable_timestamps)
                if result:
                    st.session_state['result'] = result
                    st.session_state['segments'] = segments
                    st.experimental_rerun()
            else:
                st.error("❌ 音声ファイルを選択してください")

    with col2:
        st.markdown("## 🎤 リアルタイム録音")
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎙️ マイク録音機能</h4>
            <p>下のボタンを押してマイクで直接録音し、すぐに文字起こしができます</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Streamlitの音声録音機能
        audio_value = st.audio_input("🎤 録音ボタンを押して話しかけてください")
        
        if audio_value is not None:
            st.success("✅ 録音完了！下のボタンで文字起こしを開始できます")
            
            if st.button("🔍 録音音声を文字起こし", use_container_width=True, type="secondary"):
                result, segments = transcribe_audio(audio_value, model_size, language, enable_timestamps, is_recording=True)
                if result:
                    st.session_state['result'] = result
                    st.session_state['segments'] = segments
                    st.experimental_rerun()

    # 結果表示エリア
    st.markdown("---")
    
    # セッション状態から結果を取得
    if 'result' in st.session_state and st.session_state['result']:
        display_results(st.session_state['result'], st.session_state.get('segments'))
    else:
        st.markdown("## 📝 結果表示エリア")
        st.markdown("""
        <div class="result-box">
            <div style="text-align: center; color: #999; padding: 2rem;">
                <h3>🎯 音声文字起こしを開始してください</h3>
                <p>📁 ファイルをアップロードするか、🎤 マイクで録音して文字起こしボタンを押してください</p>
                <br>
                <p><strong>✨ このツールでできること:</strong></p>
                <p>📝 高精度な音声認識 | ⏰ 時間区切り表示 | 🌍 多言語対応 | 💾 結果ダウンロード</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 <strong>AI音声文字起こしツール</strong> - Powered by OpenAI Whisper & Streamlit</p>
        <p>📧 質問・要望がございましたら、お気軽にお問い合わせください</p>
    </div>
    """, unsafe_allow_html=True)

# アプリケーション実行
if __name__ == "__main__":
    main()

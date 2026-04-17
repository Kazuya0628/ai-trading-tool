# デプロイ・セットアップ

## 必要なAPIキー

1. **Twelve Data API**（現在使用中）: https://twelvedata.com/register で無料登録
2. **Google Gemini API**: https://aistudio.google.com/ で取得（無料枠: 15 RPM / 100万トークン/日）
3. **OANDA API**（将来用）: ライブ口座開設後にAPIトークンを取得

## .env 設定例

```bash
DATA_SOURCE=twelvedata          # twelvedata または oanda
TWELVEDATA_API_KEY=your_key     # Twelve Data APIキー
GEMINI_API_KEY=your_key         # Gemini APIキー
TRADING_MODE=paper              # paper（ペーパートレード）

# メール通知（Gmail例）
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password  # Googleアプリパスワード
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=your_email@gmail.com
```

## 初回セットアップ

```bash
pip install -r requirements.txt
cp .env.example .env
# .env を編集してAPIキーを設定
```

## macOS 常駐インストール

```bash
chmod +x deploy/install_mac.sh
./deploy/install_mac.sh
nano ~/.ai-fx-bot/.env                                          # APIキー設定
launchctl load ~/Library/LaunchAgents/com.ai-fx-bot.plist       # 起動
launchctl unload ~/Library/LaunchAgents/com.ai-fx-bot.plist     # 停止
tail -f ~/.ai-fx-bot/logs/trading.log                           # ログ監視
```

## Linux 常駐インストール（systemd）

```bash
chmod +x deploy/install.sh
sudo ./deploy/install.sh
sudo nano /opt/ai-fx-bot/.env          # APIキー設定
sudo systemctl start ai-fx-bot         # 起動
sudo systemctl stop ai-fx-bot          # 停止
journalctl -u ai-fx-bot -f             # ログ監視
```

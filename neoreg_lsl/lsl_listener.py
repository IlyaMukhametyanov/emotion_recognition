from pylsl import StreamInlet, resolve_streams
import time

def connect_to_lsl(target_name):
    """Ищет поток по имени и подключается к нему."""
    print(f"🔍 Ищем поток с именем '{target_name}'...")
    streams = resolve_streams()
    target_stream = None
    for stream in streams:
        if stream.name() == target_name:
            target_stream = stream
            break

    if target_stream is None:
        print(f"❌ Поток '{target_name}' не найден.")
        return None

    print(f"✅ Подключено к потоку: {target_stream.name()}")
    inlet = StreamInlet(target_stream)
    return inlet


def record_eeg_to_file(inlet, filename="eeg_data.txt", duration=100):
    """Записывает данные EEG в текстовый файл в течение duration секунд."""
    print(f"📄 Запись данных в файл '{filename}'...")
    start_time = time.time()

    with open(filename, 'w') as f:
        # Заголовок
        f.write("timestamp\t" + "\t".join([f"ch{i + 1}" for i in range(inlet.info().channel_count())]) + "\n")

        while time.time() - start_time < duration:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample is not None:
                line = f"{timestamp:.6f}\t" + "\t".join([f"{val:.6f}" for val in sample]) + "\n"
                f.write(line)

    print("✅ Запись завершена.")
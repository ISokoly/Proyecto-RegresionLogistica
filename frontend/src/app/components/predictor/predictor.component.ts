import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-predictor',
  templateUrl: './predictor.component.html',
  imports:[CommonModule, FormsModule],
  styleUrls: ['./predictor.component.scss']
})
export class PredictorComponent {
  file: File | null = null;
  resultados: any[] = [];

  constructor(private http: HttpClient) {}

  onFileSelected(event: any) {
    this.file = event.target.files[0];
  }

  subirArchivo() {
    if (!this.file) return;

    const formData = new FormData();
    formData.append('file', this.file);

    this.http.post<any>('http://localhost:8000/predict_excel', formData).subscribe({
      next: (res) => this.resultados = res.predicciones,
      error: (err) => alert(err.error.detail || 'Error al enviar archivo')
    });
  }
}
